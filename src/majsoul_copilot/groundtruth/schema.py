"""把雀魂的 ``liqi.json`` 轉成 Python 可用的 protobuf 型別。

雀魂散布的協定定義是 **protobuf.js 的 JSON descriptor 格式**,不是 Python
``google.protobuf`` 認得的 ``FileDescriptorProto``。本模組在執行期做轉換:
走過 JSON 樹 → 組出 ``FileDescriptorProto`` → 註冊進 ``DescriptorPool`` →
產生訊息類別。好處是不需要 protoc、不需要編譯步驟,換版本只要換一個 JSON 檔。

liqi.json 的結構相當單純(實測 v0.11.243.w:1318 個訊息、1 個 enum、
3 個 service、4625 個欄位),而且**沒有** map 欄位、oneof 或 group,
所以轉換只需處理:巢狀訊息、enum、``repeated`` 標記,以及型別名稱解析。

取得與更新 liqi.json 請用 ``python tools/fetch_liqi.py``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

from majsoul_copilot.utils.paths import ASSETS_DIR

__all__ = ["DEFAULT_LIQI_PATH", "LiqiSchema", "MethodSignature"]

DEFAULT_LIQI_PATH: Path = ASSETS_DIR / "proto" / "liqi.json"

_F = descriptor_pb2.FieldDescriptorProto

_SCALAR_TYPES: dict[str, _F.Type.ValueType] = {
    "double": _F.TYPE_DOUBLE,
    "float": _F.TYPE_FLOAT,
    "int32": _F.TYPE_INT32,
    "int64": _F.TYPE_INT64,
    "uint32": _F.TYPE_UINT32,
    "uint64": _F.TYPE_UINT64,
    "sint32": _F.TYPE_SINT32,
    "sint64": _F.TYPE_SINT64,
    "fixed32": _F.TYPE_FIXED32,
    "fixed64": _F.TYPE_FIXED64,
    "sfixed32": _F.TYPE_SFIXED32,
    "sfixed64": _F.TYPE_SFIXED64,
    "bool": _F.TYPE_BOOL,
    "string": _F.TYPE_STRING,
    "bytes": _F.TYPE_BYTES,
}


class SchemaError(Exception):
    """liqi.json 的內容不是預期的結構。"""


@dataclass(frozen=True, slots=True)
class MethodSignature:
    """一個 RPC 方法的請求/回應型別。

    Wrapper 只在**請求**裡帶方法名(形如 ``.lq.Lobby.oauth2Login``),回應的
    ``Wrapper.name`` 是空的 —— 客戶端靠訊息 ID 配對。因此解析回應時,一定要
    先記住同一個 ID 的請求用了哪個方法,才知道該用哪個型別去解。
    """

    method: str  # 完整方法名,含前導點
    request_type: str  # 完整訊息名,例如 "lq.ReqOauth2Login"
    response_type: str


def _is_message(node: dict[str, Any]) -> bool:
    return "fields" in node


def _is_enum(node: dict[str, Any]) -> bool:
    return "values" in node


def _is_service(node: dict[str, Any]) -> bool:
    return "methods" in node


class LiqiSchema:
    """載入並轉換 liqi.json,提供訊息類別查詢。"""

    def __init__(self, definition: dict[str, Any], *, file_name: str = "liqi.proto") -> None:
        root = definition.get("nested")
        if not isinstance(root, dict):
            raise SchemaError("liqi.json 最外層必須有 'nested' 物件")

        self._package, types = self._split_package(root)
        self._kinds = self._collect_kinds(types, self._package)
        self._methods = self._collect_methods(types, self._package)

        file_proto = descriptor_pb2.FileDescriptorProto(
            name=file_name, package=self._package, syntax="proto3"
        )
        self._build_top_level(file_proto, types)

        self._pool = descriptor_pool.DescriptorPool()
        self._pool.Add(file_proto)
        self._class_cache: dict[str, type[Message]] = {}

    # ------------------------------------------------------------------ 建構

    @classmethod
    def load(cls, path: Path | str | None = None) -> LiqiSchema:
        """從檔案載入。``path`` 省略時使用專案內附的 ``assets/proto/liqi.json``。"""
        target = Path(path) if path is not None else DEFAULT_LIQI_PATH
        if not target.is_file():
            raise FileNotFoundError(
                f"找不到 liqi.json: {target}\n"
                "  請執行 `python tools/fetch_liqi.py` 從雀魂官方 CDN 取得。"
            )
        with target.open("r", encoding="utf-8") as fh:
            return cls(json.load(fh), file_name=target.name.replace(".json", ".proto"))

    @staticmethod
    def _is_namespace(node: dict[str, Any]) -> bool:
        """純命名空間:有 nested,但自己不是 message / enum / service。

        注意不能寫成 ``node.keys() == {"nested"}`` —— 實際的 liqi.json 裡
        ``lq`` 節點還帶了一個 ``options``(protobuf 的 file options)。
        """
        return "nested" in node and not (_is_message(node) or _is_enum(node) or _is_service(node))

    @classmethod
    def _split_package(cls, root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """把最外層的純命名空間節點抽出來當 protobuf package。

        liqi.json 的最外層是 ``{"lq": {"options": {...}, "nested": {...}}}``,
        ``lq`` 是命名空間而非訊息 → package = "lq"。
        """
        parts: list[str] = []
        current = root
        while len(current) == 1:
            name, node = next(iter(current.items()))
            if not cls._is_namespace(node):
                break
            parts.append(name)
            current = node["nested"]
        return ".".join(parts), current

    @staticmethod
    def _collect_kinds(types: dict[str, Any], package: str) -> dict[str, str]:
        """建立「完整限定名 → message / enum」對照表,供型別名稱解析使用。"""
        kinds: dict[str, str] = {}

        def walk(namespace: dict[str, Any], prefix: str) -> None:
            for name, node in namespace.items():
                fq = f"{prefix}.{name}" if prefix else name
                if _is_enum(node):
                    kinds[fq] = "enum"
                elif _is_message(node):
                    kinds[fq] = "message"
                if "nested" in node:
                    walk(node["nested"], fq)

        walk(types, package)
        return kinds

    @staticmethod
    def _collect_methods(types: dict[str, Any], package: str) -> dict[str, MethodSignature]:
        methods: dict[str, MethodSignature] = {}
        prefix = f"{package}." if package else ""
        for service_name, node in types.items():
            if not _is_service(node):
                continue
            for method_name, spec in node["methods"].items():
                # 協定裡的方法名帶前導點,例如 ".lq.Lobby.oauth2Login"
                full = f".{prefix}{service_name}.{method_name}"
                methods[full] = MethodSignature(
                    method=full,
                    request_type=f"{prefix}{spec['requestType']}",
                    response_type=f"{prefix}{spec['responseType']}",
                )
        return methods

    def _resolve(self, type_name: str, scope: str) -> str:
        """依 protobuf.js 的規則解析型別名:由內層 scope 往外層逐級嘗試。"""
        parts = scope.split(".") if scope else []
        while True:
            candidate = ".".join([*parts, type_name]) if parts else type_name
            if candidate in self._kinds:
                return candidate
            if not parts:
                raise SchemaError(f"無法解析型別 {type_name!r}(scope={scope!r})")
            parts.pop()

    def _build_message(
        self, name: str, node: dict[str, Any], scope: str
    ) -> descriptor_pb2.DescriptorProto:
        proto = descriptor_pb2.DescriptorProto(name=name)
        for field_name, field in node["fields"].items():
            descriptor = proto.field.add(
                name=field_name,
                number=field["id"],
                # liqi.json 只用到 repeated;無 rule 即 proto3 的 singular。
                label=_F.LABEL_REPEATED if field.get("rule") == "repeated" else _F.LABEL_OPTIONAL,
            )
            type_name = field["type"]
            if type_name in _SCALAR_TYPES:
                descriptor.type = _SCALAR_TYPES[type_name]
            else:
                fq = self._resolve(type_name, scope)
                descriptor.type = _F.TYPE_ENUM if self._kinds[fq] == "enum" else _F.TYPE_MESSAGE
                descriptor.type_name = f".{fq}"

        for nested_name, nested in node.get("nested", {}).items():
            nested_scope = f"{scope}.{nested_name}"
            if _is_message(nested):
                proto.nested_type.append(self._build_message(nested_name, nested, nested_scope))
            elif _is_enum(nested):
                enum = proto.enum_type.add(name=nested_name)
                for value_name, value in nested["values"].items():
                    enum.value.add(name=value_name, number=value)
        return proto

    def _build_top_level(
        self, file_proto: descriptor_pb2.FileDescriptorProto, types: dict[str, Any]
    ) -> None:
        for name, node in types.items():
            scope = f"{self._package}.{name}" if self._package else name
            if _is_message(node):
                file_proto.message_type.append(self._build_message(name, node, scope))
            elif _is_enum(node):
                enum = file_proto.enum_type.add(name=name)
                for value_name, value in node["values"].items():
                    enum.value.add(name=value_name, number=value)
            # service 不需要 descriptor,方法簽章已收在 _collect_methods

    # ------------------------------------------------------------------ 查詢

    @property
    def package(self) -> str:
        return self._package

    @property
    def methods(self) -> dict[str, MethodSignature]:
        """完整方法名(含前導點)→ 請求/回應型別。"""
        return self._methods

    def message_class(self, full_name: str) -> type[Message]:
        """取得訊息類別。

        Args:
            full_name: 完整訊息名。可帶前導點(協定中的寫法,如
                ``".lq.ActionDiscardTile"``),也可不帶(``"lq.ActionDiscardTile"``)。
        """
        key = full_name.lstrip(".")
        cached = self._class_cache.get(key)
        if cached is not None:
            return cached
        try:
            descriptor = self._pool.FindMessageTypeByName(key)
        except KeyError as exc:
            raise SchemaError(f"協定中沒有訊息型別 {full_name!r}") from exc
        cls = message_factory.GetMessageClass(descriptor)
        self._class_cache[key] = cls
        return cls

    def parse(self, full_name: str, payload: bytes) -> Any:
        """依型別名解析 protobuf 位元組。

        回傳型別標成 Any 而非 Message:訊息類別是執行期從 liqi.json 產生的,
        靜態型別檢查無從得知它有哪些欄位。
        """
        message = self.message_class(full_name)()
        message.ParseFromString(payload)
        return message

    def has_message(self, full_name: str) -> bool:
        return self._kinds.get(full_name.lstrip("."), "") == "message"

    def __repr__(self) -> str:
        messages = sum(1 for k in self._kinds.values() if k == "message")
        return (
            f"<LiqiSchema package={self._package!r} "
            f"messages={messages} methods={len(self._methods)}>"
        )
