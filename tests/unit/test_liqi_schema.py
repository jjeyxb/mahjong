"""liqi.json → Python protobuf 型別的轉換測試。

大部分用手寫的小型 schema,這樣測試不依賴外部檔案、失敗訊息也好讀;
最後一個 class 才拿專案內附的真實 liqi.json 做整體檢查。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from majsoul_copilot.groundtruth.schema import DEFAULT_LIQI_PATH, LiqiSchema, SchemaError

MINIMAL = {
    "nested": {
        "demo": {
            "options": {"java_package": "無關緊要"},
            "nested": {
                "Wrapper": {
                    "fields": {
                        "name": {"type": "string", "id": 1},
                        "data": {"type": "bytes", "id": 2},
                    }
                },
                "Tile": {
                    "fields": {
                        "code": {"type": "string", "id": 1},
                        "count": {"type": "uint32", "id": 2},
                    }
                },
                "Hand": {
                    "fields": {
                        "seat": {"type": "uint32", "id": 1},
                        "tiles": {"rule": "repeated", "type": "Tile", "id": 2},
                        "phase": {"type": "Phase", "id": 3},
                    },
                    "nested": {"Phase": {"values": {"DRAW": 0, "DISCARD": 1}}},
                },
                "Svc": {
                    "methods": {
                        "doThing": {"requestType": "Tile", "responseType": "Hand"},
                    }
                },
            },
        }
    }
}


@pytest.fixture
def schema() -> LiqiSchema:
    return LiqiSchema(MINIMAL)


class TestPackageDetection:
    def test_namespace_becomes_package(self, schema: LiqiSchema) -> None:
        assert schema.package == "demo"

    def test_options_key_does_not_break_namespace_detection(self, schema: LiqiSchema) -> None:
        """迴歸測試:真實 liqi.json 的 ``lq`` 節點同時有 options 與 nested。

        最初把命名空間判準寫成 ``keys() == {"nested"}``,結果 package 抓成空字串,
        整個 pool 是空的、所有查詢都失敗。判準必須是「沒有 fields/values/methods」。
        """
        assert schema.package == "demo"
        assert schema.has_message("demo.Wrapper")

    def test_no_namespace_yields_empty_package(self) -> None:
        flat = {"nested": {"A": {"fields": {"x": {"type": "uint32", "id": 1}}}}}
        assert LiqiSchema(flat).package == ""

    def test_missing_nested_rejected(self) -> None:
        with pytest.raises(SchemaError, match="nested"):
            LiqiSchema({"foo": 1})


class TestMessageConstruction:
    def test_scalar_roundtrip(self, schema: LiqiSchema) -> None:
        cls = schema.message_class("demo.Tile")
        restored = schema.parse("demo.Tile", cls(code="5m", count=3).SerializeToString())
        assert (restored.code, restored.count) == ("5m", 3)

    def test_repeated_and_nested_message(self, schema: LiqiSchema) -> None:
        tile_cls = schema.message_class("demo.Tile")
        hand_cls = schema.message_class("demo.Hand")
        hand = hand_cls(seat=2, tiles=[tile_cls(code="1p", count=1), tile_cls(code="2p", count=2)])
        restored = schema.parse("demo.Hand", hand.SerializeToString())
        assert restored.seat == 2
        assert [t.code for t in restored.tiles] == ["1p", "2p"]

    def test_nested_enum_resolves(self, schema: LiqiSchema) -> None:
        hand_cls = schema.message_class("demo.Hand")
        restored = schema.parse("demo.Hand", hand_cls(phase=1).SerializeToString())
        assert restored.phase == 1

    def test_leading_dot_accepted(self, schema: LiqiSchema) -> None:
        """協定裡的型別名帶前導點,查詢時不該還要呼叫端自己去掉。"""
        assert schema.message_class(".demo.Tile") is schema.message_class("demo.Tile")

    def test_class_is_cached(self, schema: LiqiSchema) -> None:
        assert schema.message_class("demo.Tile") is schema.message_class("demo.Tile")

    def test_unknown_message_raises(self, schema: LiqiSchema) -> None:
        with pytest.raises(SchemaError, match="沒有訊息型別"):
            schema.message_class("demo.NotThere")

    def test_has_message(self, schema: LiqiSchema) -> None:
        assert schema.has_message(".demo.Hand")
        assert not schema.has_message("demo.Nope")
        assert not schema.has_message("demo.Hand.Phase"), "enum 不是 message"


class TestServiceMethods:
    def test_method_signature_collected(self, schema: LiqiSchema) -> None:
        signature = schema.methods[".demo.Svc.doThing"]
        assert signature.request_type == "demo.Tile"
        assert signature.response_type == "demo.Hand"

    def test_services_are_not_messages(self, schema: LiqiSchema) -> None:
        assert not schema.has_message("demo.Svc")


class TestTypeResolution:
    def test_inner_scope_wins_over_outer(self) -> None:
        """同名型別同時存在於內外層時,應解析到最內層的那個 —— protobuf.js 的規則。"""
        definition = {
            "nested": {
                "Dup": {"fields": {"outer": {"type": "uint32", "id": 1}}},
                "Holder": {
                    "fields": {"value": {"type": "Dup", "id": 1}},
                    "nested": {"Dup": {"fields": {"inner": {"type": "string", "id": 1}}}},
                },
            }
        }
        s = LiqiSchema(definition)
        holder = s.message_class("Holder")(value={"inner": "命中內層"})
        assert s.parse("Holder", holder.SerializeToString()).value.inner == "命中內層"

    def test_unresolvable_type_raises(self) -> None:
        definition = {"nested": {"A": {"fields": {"x": {"type": "Ghost", "id": 1}}}}}
        with pytest.raises(SchemaError, match="無法解析型別"):
            LiqiSchema(definition)


class TestVendoredLiqi:
    """對專案內附的真實 liqi.json 做整體檢查。"""

    @pytest.fixture(scope="class")
    @classmethod
    def real(cls) -> LiqiSchema:
        if not DEFAULT_LIQI_PATH.is_file():
            pytest.skip("assets/proto/liqi.json 不存在,請先執行 tools/fetch_liqi.py")
        return LiqiSchema.load()

    def test_package_is_lq(self, real: LiqiSchema) -> None:
        assert real.package == "lq"

    def test_core_messages_present(self, real: LiqiSchema) -> None:
        for name in ("lq.Wrapper", "lq.ActionPrototype", "lq.ActionNewRound",
                     "lq.ActionDealTile", "lq.ActionDiscardTile", "lq.ActionChiPengGang",
                     "lq.ActionHule", "lq.ActionNoTile"):
            assert real.has_message(name), name

    def test_action_prototype_shape(self, real: LiqiSchema) -> None:
        """MJAI 轉換完全依賴這三個欄位,改了就要知道。"""
        fields = {f.name for f in real.message_class("lq.ActionPrototype").DESCRIPTOR.fields}
        assert fields == {"step", "name", "data"}

    def test_every_field_type_resolves(self, real: LiqiSchema) -> None:
        """建構過程若有任何型別解析不了會直接拋例外,能走到這裡就代表全部通過。"""
        assert len(real.methods) > 100

    def test_vendored_file_matches_recorded_hash(self) -> None:
        """內附的 liqi.json 與它的版本檔要一致,免得有人換了檔卻忘了更新版本資訊。"""
        import hashlib

        version_path = DEFAULT_LIQI_PATH.with_name("liqi.version.json")
        if not (DEFAULT_LIQI_PATH.is_file() and version_path.is_file()):
            pytest.skip("缺少 liqi.json 或其版本檔")
        meta = json.loads(Path(version_path).read_text(encoding="utf-8"))
        digest = hashlib.sha256(DEFAULT_LIQI_PATH.read_bytes()).hexdigest()
        assert digest == meta["sha256"], "liqi.json 內容與 liqi.version.json 記錄的雜湊不符"
