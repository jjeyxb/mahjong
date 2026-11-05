"""第 1 步:錄製逐幀觀測(observations.ndjson)。只存模型原始輸出,不做判斷。

=== 用法(在 MJAI 資料夾裡執行)===
    python live_v7\\record_obs.py                     自動找雀魂視窗(同 4_tsumogiri)
    python live_v7\\record_obs.py --hwnd 12345
    python live_v7\\record_obs.py --save-video        同時存原始畫面(之後可重新推論)
    python live_v7\\record_obs.py --fsm               邊錄邊跑狀態機,終端印事件
    python live_v7\\record_obs.py --video runs\\X\\frames.mp4   對錄好的影片重新推論

產物在 runs\\<執行編號>\\:
    observations.ndjson   一幀一行(格式見 obs.py 與 README)
    frames.mp4 + video_index.csv   (--save-video)原始畫面與每一幀的 t_ms
    record_summary.txt    fps、各模型推論耗時

之後用 replay.py 離線跑判斷:
    python live_v7\\replay.py runs\\<執行編號>\\observations.ndjson

=== 每個模型多久跑一次(fps 是預算)===
    局勢模型   每幀(摸入牌、手臂、宣告橫幅都在這裡,錯過一幀就可能錯過判斷點)
    牌河模型   每幀
    手牌模型   --hand-every(預設 2)  P0 的切法與摸牌牌面
    鳴牌模型   --meld-every(預設 3;0 = 不跑)  第 4 步的 consumed 要用
    場況       --ui-every(預設 5)  樣板比對,很便宜
    寶牌       --dora-every(預設 5;0 = 不跑)  裁寶牌面板用手牌模型讀(dora_reader.py);
               --dora-model tile 可換回舊的整張畫面 tile_model
沒跑的那一幀,該欄位寫 null(不是 [])。

=== 門檻 ===
局勢模型用 --region-floor(預設 0.10)把低分偵測也記下來;各類別真正的門檻
在 replay 那一側套(event_zones.ACT_CONF、drawn.DRAWN_CONF、ARM_CONF)。
這樣門檻要改時不用重錄。
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import _paths  # noqa: F401
from obs import SEATS, det_region, det_tile, dumps, new_frame

GROW_LOOKAHEAD, GROW_MARGIN, GROW_MAX = 2, 10, 200     # 同第三/四步


def find_hwnd(args):
    from capture import resolve_target_window
    if args.hwnd:
        return resolve_target_window(hwnd=args.hwnd)
    spec = importlib.util.spec_from_file_location(
        "_cap1", os.path.join(_paths.MJAI_DIR, "1_capture.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.find_window(args.window)


class VideoSource:
    """錄好的影片當輸入。t_ms 優先用 --save-video 一起存的 video_index.csv。"""

    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"❌ 開不了影片 {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        self.times = {}
        idx = os.path.join(os.path.dirname(os.path.abspath(path)),
                           "video_index.csv")
        if os.path.isfile(idx):
            with open(idx, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.times[int(r["video_frame"])] = float(r["t_ms"])
            print(f"✅ 用 video_index.csv 的時間戳({len(self.times)} 幀)")
        self.i = 0
        self.aspect_ok = True

    def grab(self):
        ok, fr = self.cap.read()
        if not ok:
            return None, None
        self.i += 1
        h, w = fr.shape[:2]
        if abs(w / h - 16 / 9) > 0.03:
            self.aspect_ok = False
        if (w, h) != (1920, 1080):
            fr = self.cv2.resize(fr, (1920, 1080),
                                 interpolation=self.cv2.INTER_AREA)
        return fr, self.times.get(self.i, (self.i - 1) * 1000.0 / self.fps)


class _NoHooks:
    """agent.py(字幕列)用的掛勾;命令列執行時全部什麼都不做。"""
    def status(self, text, kind="info"): pass
    def events(self, evs, pipe): pass
    def reply(self, trigger, reply): pass
    def log(self, text): pass
    def fps(self, fps): pass
    def frame(self, img): pass
    def window(self, hwnd, rect_fn=None): pass
    def hand(self, tiles, drawn_box): pass
    def should_stop(self): return False


def main(argv=None, hooks=None):
    hk = hooks or _NoHooks()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--hwnd", type=int)
    ap.add_argument("--window", default="雀魂")
    ap.add_argument("--video", help="對錄好的影片推論(不抓視窗)")
    ap.add_argument("--seconds", type=float, default=0.0, help="錄幾秒(0=直到按 q 或 Ctrl+C)")
    ap.add_argument("--out", help="輸出資料夾(預設 runs\\<執行編號>)")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--video-fps", type=float, default=10.0,
                    help="存影片時寫進檔頭的名目 fps(實際時間看 video_index.csv)")
    ap.add_argument("--hand-every", type=int, default=2)
    ap.add_argument("--meld-every", type=int, default=3)
    ap.add_argument("--ui-every", type=int, default=5)
    ap.add_argument("--dora-every", type=int, default=5)
    ap.add_argument("--dora-model", choices=("hand", "tile"), default="hand",
                    help="hand = 裁寶牌面板用手牌模型讀(預設,2026-11-01);"
                         "tile = 舊的整張畫面 tile_model")
    ap.add_argument("--old-dir", default=_paths.DEFAULT_OLD_DIR,
                    help="場況樣板的位置(預設 專案\\assets)")
    ap.add_argument("--region-floor", type=float, default=0.10)
    ap.add_argument("--region-model", default=None,
                    help="局勢判斷模型權重(預設 專案\\Model\\region_model_best.pt)")
    ap.add_argument("--region-imgsz", type=int, default=None,
                    help="局勢判斷模型的輸入尺寸,要跟訓練時一樣(640 / 960)")
    ap.add_argument("--river-conf", type=float, default=None)
    ap.add_argument("--no-auto-grow", action="store_true")
    ap.add_argument("--fsm", action="store_true", help="邊錄邊跑狀態機並印事件")
    ap.add_argument("--mortal", help="即時送給 Mortal(隱含 --fsm)。寫 docker 就是"
                    "預設指令(掛載 專案\\Model);也可以給完整指令,"
                    "{player_id}、{model_dir} 會被替換")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args(argv)

    # Mac:讓 YOLO 用 Apple GPU(MPS);Windows 沒有影響(CUDA 本來就會自己用)
    try:
        import platform_compat
        dev = platform_compat.setup_device()
        print(f"辨識裝置:{platform_compat.device_label(dev)}")
    except Exception as e:
        print(f"⚠️ 選擇辨識裝置失敗({e}),照 ultralytics 預設")
    import cv2
    from region_model import RegionModel
    from river import RIVER_ZONES, grow_zone, needed_growth, \
        sort_by_discard_order
    from river_model import DEFAULT_CONF as RIVER_CONF
    from river_model import RiverModel

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.out or os.path.join(_paths.RUNS_DIR, run_id)
    os.makedirs(out, exist_ok=True)

    # --- 輸入 ---------------------------------------------------------
    if a.video:
        src = VideoSource(a.video)
        grab = src.grab
        cap = src
    else:
        from capture import WindowCapture
        hwnd = find_hwnd(a)
        if hwnd is None:
            hk.status("找不到雀魂視窗", "error")
            return 2
        cap = WindowCapture(hwnd)
        # 字幕列要把手牌上的紅點畫在遊戲畫面上(網頁版:瀏覽器裡的那一塊)
        hk.window(hwnd, cap.game_screen_rect)
        t0 = time.time()

        def grab():
            fr = cap.grab()
            return fr, (time.time() - t0) * 1000.0
        first, _ = grab()
        if first is None:
            print("❌ 第一幀就抓不到。視窗是不是最小化了?")
            hk.status("抓不到遊戲畫面(視窗是不是最小化了?)", "error")
            return 2
        if not cap.aspect_ok:
            print("⚠️ 寬高比不是 16:9,也還沒在視窗裡找到遊戲畫面 —— "
                  "網頁版進到牌桌後會自動找到;Steam 版請把視窗調成 16:9。")
            hk.status("還沒找到 16:9 的遊戲畫面(網頁版進到牌桌後會自動找到)",
                      "warn")

    # --- 模型 ---------------------------------------------------------
    hk.status("載入辨識模型中…")
    region = RegionModel(model_path=a.region_model, default_conf=a.region_floor,
                         imgsz=a.region_imgsz)
    river_model = RiverModel(conf=a.river_conf or RIVER_CONF)
    hand_model = meld_model = rawui = None
    if a.hand_every > 0:
        from hand_model import HandModel
        hand_model = HandModel(region=region)
    if a.meld_every > 0:
        try:
            from meld_model import MeldModel
            meld_model = MeldModel()
        except (FileNotFoundError, SystemExit) as e:
            print(f"⚠️ 鳴牌模型載不起來,這次不錄鳴牌區:{e}")
    if a.ui_every > 0:
        tile_model = None
        if a.dora_every > 0 and a.dora_model == "hand":
            try:
                from dora_reader import DoraReader
                tile_model = DoraReader()
            except FileNotFoundError as e:
                print(f"⚠️ {e},不讀寶牌")
        elif a.dora_every > 0:
            p = os.path.join(_paths.MODEL_DIR, "tile_model_best.pt")
            if os.path.isfile(p):
                tile_model = _load_tile_model(p)
            else:
                print(f"⚠️ 找不到 {p},不讀寶牌")
        try:
            from kyoku_info import RawUiReader
            rawui = RawUiReader(a.old_dir, tile_model)
        except Exception as e:      # 樣板資料夾不在 -> 不讀場況,但照錄
            print(f"⚠️ 場況讀取器載不起來,這次不錄場況:{e}")

    import drawn_lib
    tile_lib = drawn_lib.DrawnLibrary(drawn_lib.DEFAULT_LIB)
    print(f"✅ 摸入牌樣板庫預載 {sum(len(v) for v in tile_lib.lib.values())} 張")

    pipe = guard = sink = None
    if a.fsm or a.mortal:
        from mortal_guard import FileSink, MortalGuard, MultiSink, ProcessSink
        from pipeline import Pipeline, describe
        psink = ProcessSink(a.mortal, os.path.join(
            out, "mortal_replies.jsonl"),
            on_reply=lambda s: (print(s), hk.log(s))) if a.mortal else None
        if psink is not None:
            psink.on_reply_obj = hk.reply
        sink = MultiSink(FileSink(os.path.join(out, "mjai.jsonl")), psink)
        guard = MortalGuard(sink, log=lambda s: (print(s), hk.log(s)))
        pipe = Pipeline(run_id, guard=guard)

    # --- 輸出 ---------------------------------------------------------
    fh = open(os.path.join(out, "observations.ndjson"), "a",
              encoding="utf-8")
    vw = vidx = None
    if a.save_video:
        vw = cv2.VideoWriter(os.path.join(out, "frames.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), a.video_fps,
                             (1920, 1080))
        vidx = open(os.path.join(out, "video_index.csv"), "w", newline="",
                    encoding="utf-8")
        vidx.write("video_frame,frame_index,t_ms\n")

    extra = {s: 0 for s in SEATS}
    ms = defaultdict(list)
    frame_no = 0
    t_start = time.time()
    last_print = t_start
    print(f"\n🎬 {run_id} 錄製中 -> {out}")
    print(f"   手牌每 {a.hand_every} 幀、鳴牌每 {a.meld_every} 幀、"
          f"場況每 {a.ui_every} 幀、寶牌每 {a.dora_every} 幀(0 = 不跑)")
    print("   " + ("q 結束" if not a.no_preview else "Ctrl+C 結束") + "\n")
    hk.status("ready", "ready")
    ev_fh = open(os.path.join(out, "events_live.ndjson"), "a",
                 encoding="utf-8") if pipe is not None else None
    none_run = 0
    try:
        while True:
            if hk.should_stop():
                break
            if a.seconds and time.time() - t_start >= a.seconds:
                break
            frame, t_ms = grab()
            if frame is None:
                if a.video:
                    break
                none_run += 1
                if none_run == 30:
                    hk.status("抓不到遊戲畫面(視窗是不是最小化了?)", "warn")
                continue
            if none_run >= 30:
                hk.status("畫面恢復", "info")
            none_run = 0
            frame_no += 1
            o = new_frame(run_id, frame_no, t_ms)
            o["models"] = {"region": {"ran": True}, "river": {"ran": True},
                           "hand": {"ran": False}, "meld": {"ran": False},
                           "ui": {"ran": False}, "dora": {"ran": False}}

            ti = time.time()
            rdets = region.detect(frame)
            ms["region"].append((time.time() - ti) * 1000)
            o["region_detections"] = [det_region(d) for d in rdets]

            zones = {s: grow_zone(s, RIVER_ZONES[s], extra[s])
                     for s in SEATS}
            ti = time.time()
            tiles = river_model.detect(frame, zones=zones)
            ms["river"].append((time.time() - ti) * 1000)
            o["river"] = {s: {"zone": list(zones[s]), "inference_ok": True,
                              "tiles": [det_tile(d) for d in tiles
                                        if d["seat"] == s]}
                          for s in SEATS}
            if not a.no_auto_grow:
                for s in SEATS:
                    mine = [d for d in tiles if d["seat"] == s]
                    if not mine:
                        extra[s] = 0
                        continue
                    need = needed_growth(s, sort_by_discard_order(s, mine),
                                         zones[s], GROW_LOOKAHEAD,
                                         GROW_MARGIN)
                    if need:
                        extra[s] = min(GROW_MAX, extra[s] + need)

            if hand_model and frame_no % a.hand_every == 0:
                ti = time.time()
                # ★ 塗黑只用 0.25 以上的 meld_area(跟 5_hand.py 一樣)。錄製端
                #   的局勢偵測門檻壓到 0.10,低分的 meld_area 框直接拿去塗黑
                #   會把真的手牌也塗掉。
                hd = hand_model.detect(
                    frame, region_dets=[d for d in rdets
                                        if d["cls"] == "meld_area"
                                        and d["conf"] >= 0.25])
                ms["hand"].append((time.time() - ti) * 1000)
                o["p0_hand"] = {"tiles": [det_tile(d) for d in hd],
                                "includes_drawn": True}
                o["models"]["hand"]["ran"] = True
                # P0 摸入牌的第二意見(樣板比對,drawn_lib.py)
                ti = time.time()
                o["p0_hand"]["drawn_lib"] = drawn_lib.observe(
                    tile_lib, frame, o["p0_hand"]["tiles"], rdets)
                o["p0_hand"]["hand_fix"] = drawn_lib.observe_hand(
                    tile_lib, frame, o["p0_hand"]["tiles"], rdets)
                ms["drawn_lib"].append((time.time() - ti) * 1000)
                drawn_box = next((d["box"] for d in rdets
                                  if d["cls"] == "drawn_tile"
                                  and d["conf"] >= 0.3), None)
                hk.hand(o["p0_hand"]["tiles"], drawn_box)

            if meld_model and frame_no % a.meld_every == 0:
                ti = time.time()
                md = meld_model.detect(frame)
                ms["meld"].append((time.time() - ti) * 1000)
                o["meld"] = {s: {"tiles": [det_tile(d) for d in md
                                           if d["seat"] == s]}
                             for s in SEATS}
                o["models"]["meld"]["ran"] = True

            # 新的一局開始、場況還沒讀齊:每一幀都讀(莊家開局很快就要打第一張)
            rush = pipe is not None and pipe.start_rush(frame_no)
            if rawui and (frame_no % a.ui_every == 0 or rush):
                ti = time.time()
                want_dora = a.dora_every > 0 and \
                    (frame_no % a.dora_every == 0 or rush)
                saved = rawui.tile_model
                if not want_dora:
                    rawui.tile_model = None
                ui, dora = rawui.read(frame)
                rawui.tile_model = saved
                ms["ui"].append((time.time() - ti) * 1000)
                o["ui_context"] = ui
                o["models"]["ui"]["ran"] = True
                if dora is not None:
                    o["dora_observation"] = dora
                    o["models"]["dora"]["ran"] = True

            o["quality"] = {"aspect_ok": bool(getattr(cap, "aspect_ok",
                                                      True)),
                            "client_size": list(getattr(cap, "client_size",
                                                        None) or [])}
            fh.write(dumps(o) + "\n")
            if frame_no % 20 == 0:
                fh.flush()
            if vw is not None:
                vw.write(frame)
                vidx.write(f"{frame_no},{frame_no},{t_ms:.1f}\n")

            if pipe is not None:
                evs = pipe.feed(o)
                for ev in evs:
                    print(f"f{frame_no:05d} {t_ms / 1000:7.1f}s  "
                          f"{describe(ev)}   [{pipe.fsm.state!r}]")
                    ev_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                if evs:
                    ev_fh.flush()
                hk.events(evs, pipe)          # 每幀都叫:字幕列靠它看守門器狀態
                for m in pipe.sent:
                    print(f"      → mjai {json.dumps(m, ensure_ascii=False)}")

            now = time.time()
            if now - last_print >= 5:
                fps = frame_no / (now - t_start)
                print(f"  ... {frame_no} 幀,{fps:.1f} fps,"
                      + "  ".join(f"{k} {sum(v[-50:]) / len(v[-50:]):.0f}ms"
                                  for k, v in ms.items() if v))
                last_print = now
                hk.fps(fps)
            hk.frame(frame)
            if not a.no_preview:
                small = cv2.resize(frame, (640, 360))
                cv2.putText(small, f"REC f{frame_no}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("record_obs", small)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        try:
            tile_lib.save(os.path.join(out, "p0_tile_lib.npz"))
        except Exception as e:
            print(f"⚠️ 摸入牌樣板庫存不下來:{e}")
        if pipe is not None:
            pipe.finish()
            sink.close()
            ev_fh.close()
            # 局中事件的最終版(狀態都定案之後;跟 replay.py 輸出的格式一樣)
            with open(os.path.join(out, "events.ndjson"), "w",
                      encoding="utf-8") as efh:
                for e in pipe.ledger.ordered():
                    efh.write(json.dumps(e, ensure_ascii=False) + "\n")
                for e in pipe.ledger.events:
                    if e["type"] == "correction":
                        efh.write(json.dumps(e, ensure_ascii=False) + "\n")
            with open(os.path.join(out, "delivery.ndjson"), "w",
                      encoding="utf-8") as dfh:
                for r in guard.delivery:
                    dfh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("\n".join(guard.report()))
        if vw is not None:
            vw.release()
            vidx.close()
        if not a.no_preview:
            cv2.destroyAllWindows()

    el = max(1e-6, time.time() - t_start)
    lines = [f"run {run_id}  幀數 {frame_no}  {el:.1f}s  "
             f"平均 {frame_no / el:.2f} fps",
             "各模型平均推論耗時(ms):"]
    for k, v in ms.items():
        if v:
            s = sorted(v)
            lines.append(f"  {k:<7} 次數 {len(v):>6}  平均 {sum(v)/len(v):6.1f}"
                         f"  中位 {s[len(s)//2]:6.1f}  最慢 {s[-1]:6.1f}")
    with open(os.path.join(out, "record_summary.txt"), "w",
              encoding="utf-8") as sf:
        sf.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n下一步:python live_v7\\replay.py "
          f"{os.path.join(out, 'observations.ndjson')}")
    return 0


def _load_tile_model(path):
    """舊專案的 tile_model(整張畫面推論的牌面模型),只拿來讀寶牌。

    ★ 不 import 舊專案的 tile_model.py:那個資料夾有同名的 river.py 等,
      加進 sys.path 會蓋掉 MJAI 這邊的。這裡直接用 ultralytics 載權重,
      輸出格式跟舊的 TileModel.detect 一樣(整張畫面座標 + merge_overlaps)。
    """
    from ultralytics import YOLO
    from tiles import merge_overlaps

    class _Tile:
        def __init__(self, p, conf=0.1):
            print(f"⏳ 載入寶牌用的 tile_model:{p}")
            self.model = YOLO(p)
            self.names = dict(self.model.names)
            self.conf = conf

        def detect(self, frame):
            r = self.model.predict(frame, conf=self.conf, verbose=False)[0]
            out = []
            for b in r.boxes:
                pai = self.names[int(b.cls[0])]
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                out.append({"pai": pai, "conf": float(b.conf[0]),
                            "box": (x1, y1, x2, y2),
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2)})
            return merge_overlaps(out)

    return _Tile(path)


if __name__ == "__main__":
    sys.exit(main())
