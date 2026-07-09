from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.cubism_api import (
    APIOperation,
    ConnectionOptions,
    CubismAPISession,
    build_named_operation,
    build_request,
    execute_operation,
    parse_parameter_assignments,
)


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(payload))
        response = dict(self.responses.pop(0))
        response.setdefault("RequestId", payload["RequestId"])
        response.setdefault("Method", payload["Method"])
        return response


def test_build_request_has_stable_shape() -> None:
    payload = build_request(
        "GetCurrentModelUID",
        token="secret",
        request_id="request-1",
        timestamp_ms=123,
    )
    assert payload == {
        "Version": "0.9.1",
        "Timestamp": 123,
        "RequestId": "request-1",
        "Type": "Request",
        "Method": "GetCurrentModelUID",
        "Data": {},
        "Token": "secret",
    }


def test_get_parameters_uses_documented_version() -> None:
    operation = build_named_operation(
        "cubism_api.get_parameters",
        {"model_uid": "model-1", "use_current_model": False},
    )
    assert operation.method == "GetParameters"
    assert operation.version == "1.0.1"
    assert operation.data == {"ModelUID": "model-1"}


def test_document_snapshot_plans_composite_operation() -> None:
    operation = build_named_operation("cubism_api.get_document_snapshot")
    assert operation.method == "GetDocumentSnapshot"


def test_parse_parameter_assignments() -> None:
    assert parse_parameter_assignments(["ParamAngleX=30", "ParamAngleY=0"]) == [
        {"Id": "ParamAngleX", "Value": 30.0},
        {"Id": "ParamAngleY", "Value": 0.0},
    ]


def test_execute_operation_registers_checks_approval_and_resolves_model(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            {"Type": "Response", "Data": {"Token": "token-1"}},
            {"Type": "Response", "Data": {"Result": True}},
            {"Type": "Response", "Data": {"ModelUID": "model-1"}},
            {"Type": "Response", "Data": {"Parameters": []}},
        ]
    )
    operation = APIOperation(
        "cubism_api.get_parameters",
        "GetParameters",
        "1.0.1",
        needs_current_model=True,
    )
    options = ConnectionOptions(token_file=tmp_path / "token.json")
    result = asyncio.run(execute_operation(operation, options, transport=transport))

    assert result["status"] == "completed"
    assert [request["Method"] for request in transport.requests] == [
        "RegisterPlugin",
        "GetIsApproval",
        "GetCurrentModelUID",
        "GetParameters",
    ]
    assert transport.requests[-1]["Data"] == {"ModelUID": "model-1"}
    assert (tmp_path / "token.json").exists()


def test_execute_document_snapshot_keeps_documents_and_current_model_together(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            {"Type": "Response", "Data": {"Token": "token-1"}},
            {"Type": "Response", "Data": {"Result": True}},
            {"Type": "Response", "Data": {"ModelingDocuments": []}},
            {"Type": "Response", "Data": {"ModelUID": "model-1"}},
        ]
    )
    operation = build_named_operation("cubism_api.get_document_snapshot")
    options = ConnectionOptions(token_file=tmp_path / "token.json")
    result = asyncio.run(execute_operation(operation, options, transport=transport))
    assert result["response"] == {
        "Documents": {"ModelingDocuments": []},
        "CurrentModel": {"ModelUID": "model-1"},
    }


def test_api_session_reuses_one_registration_for_related_operations(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            {"Type": "Response", "Data": {"Token": "token-1"}},
            {"Type": "Response", "Data": {"Result": True}},
            {"Type": "Response", "Data": {"ModelingDocuments": []}},
            {"Type": "Response", "Data": {"ModelingDocuments": []}},
            {"Type": "Response", "Data": {"ModelUID": "model-1"}},
        ]
    )
    options = ConnectionOptions(token_file=tmp_path / "token.json")

    async def run() -> None:
        async with CubismAPISession(options, transport=transport) as session:
            await session.run(build_named_operation("cubism_api.get_documents"))
            await session.run(build_named_operation("cubism_api.get_document_snapshot"))

    asyncio.run(run())
    assert [request["Method"] for request in transport.requests] == [
        "RegisterPlugin",
        "GetIsApproval",
        "GetDocuments",
        "GetDocuments",
        "GetCurrentModelUID",
    ]
