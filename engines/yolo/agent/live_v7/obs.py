"""逐幀觀測(observations.ndjson)的格式:寫、讀、檢查。

依 v7 規格 `majsoul_v7_observation_fields.csv`,加上兩處修正(見 README):
  1. Detection 的欄位直接沿用舊程式的偵測 dict,下游的 RiverTracker /
     DahaiGate / DrawnWatch 不用改就能讀:
         局勢模型  {cls, conf, box, center, clipped}
         牌面模型  {pai, conf, box, center, seat}
  2. 只存模型原始輸出。「穩定」「定案」「切法」都是判斷結果,不進觀測檔 ——
     放進去的話,以後改判斷規則就得重新錄影。

★ 空 list `[]` = 模型這一幀有跑、沒偵測到。
  `null`       = 這一幀沒跑這個模型(或沒提供)。兩者不能混用。

一幀一行 JSON,寫入後不改寫。
"""
import json

SCHEMA = "v7-observation-1"
SEATS = ("P0", "P1", "P2", "P3")


def _r(v, nd=4):
    return round(float(v), nd)


def det_region(d):
    """局勢模型的偵測 -> 可序列化 dict。"""
    return {"cls": d["cls"], "conf": _r(d["conf"]),
            "box": [int(v) for v in d["box"]],
            "center": [int(v) for v in d["center"]],
            "clipped": bool(d.get("clipped", False))}


def det_tile(d):
    """牌面模型(牌河/手牌/鳴牌/寶牌)的偵測 -> 可序列化 dict。"""
    out = {"pai": d["pai"], "conf": _r(d["conf"]),
           "box": [int(v) for v in d["box"]],
           "center": [int(v) for v in d["center"]]}
    if d.get("seat") is not None:
        out["seat"] = d["seat"]
    if d.get("clipped"):
        out["clipped"] = True
    return out


def new_frame(run_id, frame_index, t_ms, frame_size=(1920, 1080)):
    """一幀的骨架。各模型欄位預設 null(= 沒跑),跑了才填。"""
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "game_id": None,
        # hand_id 由下游(pipeline)依換局判斷決定,錄製端不猜。
        "hand_id": None,
        "frame_index": int(frame_index),
        "t_ms": _r(t_ms, 1),
        "frame_size": {"width": int(frame_size[0]),
                       "height": int(frame_size[1])},
        "models": {},
        "region_detections": None,
        "river": None,
        "p0_hand": None,
        "meld": None,
        "dora_observation": None,
        "ui_context": None,
        "quality": {},
    }


def dumps(rec):
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def iter_ndjson(path):
    """逐行讀。壞掉的行(錄到一半被中斷)跳過並記下行號。"""
    bad = []
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad.append(no)
    if bad:
        print(f"⚠️ {path}: {len(bad)} 行讀不懂,已跳過(第一行 {bad[0]})")


def validate(rec):
    """回傳問題清單,空的代表格式正確。只檢查結構,不檢查內容合不合理。"""
    p = []
    if rec.get("schema_version") != SCHEMA:
        p.append(f"schema_version 不是 {SCHEMA}")
    for k in ("run_id", "frame_index", "t_ms", "frame_size", "models",
              "quality"):
        if k not in rec:
            p.append(f"缺 {k}")
    rd = rec.get("region_detections")
    if rd is not None:
        for d in rd:
            if not {"cls", "conf", "box", "center"} <= set(d):
                p.append("region_detections 有一筆缺欄位")
                break
    rv = rec.get("river")
    if rv is not None:
        for s in SEATS:
            if s not in rv:
                p.append(f"river 缺 {s}")
                continue
            if rv[s].get("tiles") is None:
                continue
            for d in rv[s]["tiles"]:
                if not {"pai", "conf", "box", "center"} <= set(d):
                    p.append(f"river.{s} 有一筆缺欄位")
                    break
    return p


def as_tuple_det(d):
    """觀測檔讀回來的 list -> 舊程式慣用的 tuple(box / center)。

    RiverTracker 用 center 算距離,box 算面積,tuple 或 list 都能算;但
    DahaiGate.note_said 存的是 tuple(center),比對時 tuple 與 list 都能用
    索引取值,所以不轉也行。轉成 tuple 是為了跟實機路徑完全一樣。
    """
    out = dict(d)
    out["box"] = tuple(d["box"])
    out["center"] = tuple(d["center"])
    return out
