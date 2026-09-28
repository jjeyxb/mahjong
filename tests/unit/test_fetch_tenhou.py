"""天鳳存檔清單的解析與篩選。

網路那一半不測(那是別人的伺服器);測的是**選錯東西也不會報錯**的那幾個地方:

* 挑錯前綴 → 抓到的是別的卓,語料整份是錯的桌次,而每一場都轉得成功。
* 日期切片算錯 → ``--since`` 靜默失效,抓回來的範圍不是你要的。
* 規則字串判斷錯 → 東風戰混進來,要等轉換那一步才一場一場失敗。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from fetch_tenhou import (
    Game,
    focus_filter,
    list_daily_archives,
    parse_archive,
    player_counts,
)

#: `?old` 清單的真實形狀(從 tenhou.net 取回來的片段)。
#: 同時含日檔與 ``.log.gz``,以及**別的卓**的前綴 —— 那才是這組測試的重點。
_OLD_LISTING = """
list = [
{file:'2025/sca20251201.log.gz',size:1},
{file:'2026/scc20260101.html.gz',size:2},
{file:'2026/sca20260102.html.gz',size:3},
{file:'2026/scc20260102.html.gz',size:4},
{file:'2026/scd20260103.html.gz',size:5},
{file:'2026/scc20260918.html.gz',size:6},
];
"""


@pytest.fixture
def listing(monkeypatch):
    """把清單換成上面那段,不碰網路。"""
    import fetch_tenhou

    monkeypatch.setattr(
        fetch_tenhou, "fetch", lambda _url, **_kw: _OLD_LISTING.encode("utf-8")
    )


# 這個 class 的每一條都靠 `listing` 把網路換掉,但沒有一條會去讀它的回傳值 ——
# 用 usefixtures 宣告依賴,而不是收一個用不到的參數。
@pytest.mark.usefixtures("listing")
class TestDailyArchives:
    def test_only_the_houou_lobby_is_taken(self) -> None:
        """只要 ``scc``。

        ``sca`` / ``scd`` 是別的卓。混進來不會有任何錯誤訊息 —— 牌譜一樣轉得成功,
        只是訓練語料變成了一堆不同水準的對局,而 M8 要學的是**鳳凰卓的打法風格**。
        """
        names = list_daily_archives(None, None)
        assert all("/scc" in n for n in names)
        assert len(names) == 3

    def test_results_are_oldest_first(self) -> None:
        names = list_daily_archives(None, None)
        assert names == sorted(names)

    def test_since_is_inclusive(self) -> None:
        assert list_daily_archives("20260102", None) == [
            "2026/scc20260102.html.gz",
            "2026/scc20260918.html.gz",
        ]

    def test_until_is_inclusive(self) -> None:
        assert list_daily_archives(None, "20260102") == [
            "2026/scc20260101.html.gz",
            "2026/scc20260102.html.gz",
        ]

    def test_the_date_is_read_from_the_filename_not_the_directory(self) -> None:
        """日期切片是 ``[3:11]`` **在去掉年份目錄之後**。

        忘了去目錄的話會切到 ``6/scc2026``,比較起來全部落在 ``--since`` 之前,
        於是 ``--since`` 靜默地什麼都篩不掉 —— 看起來像「範圍剛好全中」。
        """
        assert list_daily_archives("20260900", None) == ["2026/scc20260918.html.gz"]

    def test_an_empty_range_says_so(self) -> None:
        with pytest.raises(SystemExit, match="沒有日檔"):
            list_daily_archives("20270101", None)

    def test_a_listing_without_daily_files_says_the_format_changed(self, monkeypatch) -> None:
        import fetch_tenhou

        monkeypatch.setattr(fetch_tenhou, "fetch", lambda _url, **_kw: b"list = [];")
        with pytest.raises(SystemExit, match="格式可能改了"):
            list_daily_archives(None, None)


class TestIndexCache:
    """索引快取。

    重點不是省時間,是**別重複跟人家要同一份東西** —— 放大語料是一趟跑好幾小時
    的事,中途被打斷重跑就會再向天鳳要 261 份索引。
    """

    _HTML = (
        "00:01 | 22 | 四鳳南喰赤－ | "
        '<a href="http://tenhou.net/0/?log=2026010100gm-00a9-0000-abc">牌譜</a>\n'
        "00:02 | 22 | 四鳳東喰赤－ | "
        '<a href="http://tenhou.net/0/?log=2026010100gm-00a9-0000-def">牌譜</a>'
    )

    def test_the_second_run_does_not_hit_the_network(self, tmp_path, monkeypatch) -> None:
        import fetch_tenhou
        from fetch_tenhou import collect_games

        calls = []

        def counted(url: str, **_kw: object) -> bytes:
            calls.append(url)
            return self._HTML.encode("utf-8")

        monkeypatch.setattr(fetch_tenhou, "fetch", counted)

        first = collect_games(["2026/scc20260101.html.gz"], 0.0, tmp_path)
        assert len(calls) == 1

        second = collect_games(["2026/scc20260101.html.gz"], 0.0, tmp_path)
        assert len(calls) == 1, "第二趟又去要了一次"
        assert [(g.log_id, g.rule) for g in second] == [(g.log_id, g.rule) for g in first]

    def test_a_name_with_a_directory_becomes_one_flat_file(self, tmp_path, monkeypatch) -> None:
        """日檔的名字帶 ``2026/`` —— 直接當檔名會寫到不存在的子目錄去。"""
        import fetch_tenhou

        monkeypatch.setattr(
            fetch_tenhou, "fetch", lambda _url, **_kw: self._HTML.encode("utf-8")
        )
        from fetch_tenhou import collect_games

        collect_games(["2026/scc20260101.html.gz"], 0.0, tmp_path)
        assert (tmp_path / "2026_scc20260101.html.gz.tsv").exists()

    def test_no_cache_dir_means_no_files_written(self, tmp_path, monkeypatch) -> None:
        import fetch_tenhou
        from fetch_tenhou import collect_games

        monkeypatch.setattr(
            fetch_tenhou, "fetch", lambda _url, **_kw: self._HTML.encode("utf-8")
        )
        collect_games(["2026/scc20260101.html.gz"], 0.0, None)
        assert list(tmp_path.iterdir()) == []


class TestGameFilters:
    @pytest.mark.parametrize(
        ("rule", "four", "hanchan", "houou"),
        [
            # 實測 2026-01-01/02 兩天的 scc 存檔只出現這三種規則
            ("四鳳南喰赤－", True, True, True),
            ("三鳳南喰赤－", False, True, True),  # 三人麻將
            ("四鳳東喰赤速", True, False, True),  # 東風戰 —— Mortal 直接拒絕
            # 其他卓等:scc 裡沒有,但過濾不該靠「剛好沒有」成立
            ("四特南喰赤－", True, True, False),  # 特上卓
            ("四上南喰赤－", True, True, False),  # 上級卓
            ("四般南喰赤－", True, True, False),  # 一般卓
        ],
    )
    def test_rule_string_is_read_correctly(
        self, rule: str, four: bool, hanchan: bool, houou: bool
    ) -> None:
        game = Game(log_id="2026010100gm-00a9-0000-deadbeef", rule=rule)
        assert game.is_four_player is four
        assert game.is_hanchan is hanchan
        assert game.is_houou is houou

    def test_lower_lobbies_are_rejected_even_though_scc_has_none(self) -> None:
        """卓等要**明確**檢查,不能靠「``scc`` 剛好只有鳳凰卓」。

        原本的過濾只看四人與南場,拿到純鳳凰卓語料純粹因為前綴選對了 ——
        那是個沒寫下來、也沒被驗證的隱含相依。天鳳改了 ``scc`` 的涵蓋範圍,
        或有人把前綴打成 ``scb``,較低等級的對局就會默默混進語料:
        每一場都轉得成功、``validate_logs`` 全過,而訓出來的「風格」
        是一鍋不同水準的平均。
        """
        tokujou = Game(log_id="x", rule="四特南喰赤－")
        assert tokujou.is_four_player and tokujou.is_hanchan
        assert not tokujou.is_houou, "特上卓不該被當成鳳凰卓"

    def test_the_shard_comes_from_the_log_id(self) -> None:
        """分層是給 train/val 用 glob 切開用的,日期取自 ID 前 8 碼。"""
        game = Game(log_id="2026091900gm-00a9-0000-740c25b8", rule="四鳳南喰赤－")
        assert game.shard == ("2026", "09", "19")


class TestFocusPlayers:
    """把固定的下載預算集中在常打的人身上。

    風格統計是 per-player 的:2 萬場均勻散在幾千個玩家身上,每人只有十幾場,
    那種統計量撐不起分群。而「誰打得多」在索引列上就寫著,不必下載牌譜。
    """

    @staticmethod
    def _game(log_id: str, *names: str) -> Game:
        return Game(log_id=log_id, rule="四鳳南喰赤－", names=names)

    def test_counts_every_appearance(self) -> None:
        games = [
            self._game("a", "甲", "乙", "丙", "丁"),
            self._game("b", "甲", "乙", "戊", "己"),
            self._game("c", "甲", "庚", "辛", "壬"),
        ]
        counts = player_counts(games)
        assert counts["甲"] == 3
        assert counts["乙"] == 2
        assert counts["丙"] == 1

    def test_one_listed_player_is_enough_to_keep_a_game(self) -> None:
        """不需要四家都在名單上。

        `train.py` 把名單傳給 ``GameplayLoader``,只有名單上那些人的決策會變成
        訓練樣本,其他三家只是環境。所以一場有一個目標玩家就有價值 ——
        要求四家全中會把幾乎所有對局都丟掉。
        """
        games = [
            self._game("a", "常客", "路人1", "路人2", "路人3"),
            self._game("b", "常客", "路人4", "路人5", "路人6"),
            self._game("c", "路人7", "路人8", "路人9", "路人10"),
        ]
        focused = focus_filter(games, 1)
        assert [g.log_id for g in focused] == ["a", "b"]

    def test_chronological_order_is_preserved(self) -> None:
        """順序決定日期分層怎麼填,不能被篩選打亂。"""
        games = [self._game(f"{i}", "常客", "x", "y", "z") for i in range(5)]
        assert [g.log_id for g in focus_filter(games, 1)] == ["0", "1", "2", "3", "4"]


class TestParseArchive:
    def test_extracts_rule_and_log_id(self) -> None:
        html = (
            "00:01 | 22 | 四鳳南喰赤－ | "
            '<a href="http://tenhou.net/0/?log=2026010100gm-00a9-0000-abc">牌譜</a>'
            " | 甲(+40.5) 乙(+5.0)"
        )
        games = parse_archive(html)
        assert len(games) == 1
        assert games[0].rule == "四鳳南喰赤－"
        assert games[0].log_id == "2026010100gm-00a9-0000-abc"

    def test_rows_without_a_log_link_are_skipped(self) -> None:
        """有些列沒有牌譜連結 —— 那不是錯誤,就是沒有。"""
        assert parse_archive("00:01 | 22 | 四鳳南喰赤－ | 甲(+40.5)") == []

    def test_a_row_without_names_still_yields_a_game(self) -> None:
        """名字那一段是選擇性的。

        設成必要的話,萬一哪天有一列沒帶名字,整場會被**靜默丟掉** ——
        而牌譜本身是好的,名字只是拿來數頻率的附加資訊。
        """
        html = '00:01 | 22 | 四鳳南喰赤－ | <a href="?log=abc">牌譜</a>'
        games = parse_archive(html)
        assert len(games) == 1
        assert games[0].log_id == "abc"
        assert games[0].names == ()

    def test_names_do_not_bleed_across_lines(self) -> None:
        """`[^|]*` 不排除換行的話會吃到下一列,把別人的名字接到這一場上。

        那個錯誤不會報,只會讓玩家頻率表算錯 —— 而頻率表決定了要抓誰的對局。
        """
        html = "\n".join(
            [
                '00:01 | 22 | 四鳳南喰赤－ | <a href="?log=first">牌譜</a>',
                '00:02 | 22 | 四鳳南喰赤－ | <a href="?log=second">牌譜</a> | 乙(+1.0)',
            ]
        )
        games = parse_archive(html)
        assert [g.log_id for g in games] == ["first", "second"]
        assert games[0].names == ()
        assert games[1].names == ("乙",)
