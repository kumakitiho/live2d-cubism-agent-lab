from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import websockets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 22033
DEFAULT_APP_NAME = "live2d-cubism-agent-lab"
DEFAULT_TOKEN_FILE = Path(".live2d-agent/cubism-token.json")

METHOD_VERSIONS = {
    "RegisterPlugin": "1.0.0",
    "GetIsApproval": "0.9.0",
    "GetDocuments": "0.9.1",
    "GetCurrentModelUID": "0.9.1",
    "GetCurrentEditMode": "0.9.0",
    "GetParameters": "1.0.1",
    "GetParameterValues": "0.9.1",
    "SetParameterValues": "0.9.1",
    "ClearParameterValues": "0.9.1",
    "SendCubismLog": "0.9.3",
}


class CubismAPIError(RuntimeError):
    """Base error for Cubism API failures."""


class CubismApprovalRequired(CubismAPIError):
    """Raised when the Cubism-side Allow checkbox is not enabled."""


@dataclass(frozen=True)
class APIOperation:
    command: str
    method: str
    version: str
    data: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = True
    needs_current_model: bool = False


@dataclass(frozen=True)
class ConnectionOptions:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    app_name: str = DEFAULT_APP_NAME
    token: str | None = None
    token_file: Path = DEFAULT_TOKEN_FILE
    timeout: float = 10.0

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


class RequestTransport(Protocol):
    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...


def build_request(
    method: str,
    data: Mapping[str, Any] | None = None,
    *,
    version: str | None = None,
    token: str | None = None,
    request_id: str | None = None,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    if method not in METHOD_VERSIONS and version is None:
        raise ValueError(f"unknown Cubism API method: {method}")
    payload: dict[str, Any] = {
        "Version": version or METHOD_VERSIONS[method],
        "Timestamp": timestamp_ms if timestamp_ms is not None else int(time.time() * 1000),
        "RequestId": request_id or str(uuid.uuid4()),
        "Type": "Request",
        "Method": method,
        "Data": dict(data or {}),
    }
    if token:
        payload["Token"] = token
    return payload


class TokenStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> str | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        token = raw.get("token") if isinstance(raw, dict) else None
        return str(token) if token else None

    def save(self, token: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"token": token}, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)


class WebSocketTransport:
    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout
        self._socket: Any = None

    async def __aenter__(self) -> WebSocketTransport:
        self._socket = await websockets.connect(
            self.url,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._socket is not None:
            await self._socket.close()

    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise CubismAPIError("WebSocket transport is not connected")
        await self._socket.send(json.dumps(payload, ensure_ascii=False))
        expected_id = payload.get("RequestId")
        async with asyncio.timeout(self.timeout):
            while True:
                raw = await self._socket.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                response = json.loads(raw)
                if not isinstance(response, dict):
                    raise CubismAPIError("Cubism response must be a JSON object")
                if response.get("RequestId") == expected_id:
                    return response


class CubismClient:
    def __init__(self, transport: RequestTransport, *, token: str | None = None) -> None:
        self.transport = transport
        self.token = token

    async def _send(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self.transport.request(payload)
        response_type = response.get("Type")
        if response_type == "Error":
            data = response.get("Data")
            error_type = data.get("ErrorType") if isinstance(data, Mapping) else "Unknown"
            raise CubismAPIError(f"{response.get('Method')}: {error_type}")
        if response_type != "Response":
            raise CubismAPIError(f"unexpected response type: {response_type}")
        return response

    async def register(self, app_name: str) -> str:
        data: dict[str, Any] = {"Name": app_name}
        if self.token:
            data["Token"] = self.token
        response = await self._send(build_request("RegisterPlugin", data))
        response_data = response.get("Data")
        token = response_data.get("Token") if isinstance(response_data, Mapping) else None
        if not token:
            raise CubismAPIError("RegisterPlugin did not return a token")
        self.token = str(token)
        return self.token

    async def get_approval(self) -> bool:
        response = await self._send(
            build_request("GetIsApproval", token=self.token, version="0.9.0")
        )
        data = response.get("Data")
        return bool(data.get("Result")) if isinstance(data, Mapping) else False

    async def call(self, operation: APIOperation) -> dict[str, Any]:
        return await self._send(
            build_request(
                operation.method,
                operation.data,
                version=operation.version,
                token=self.token,
            )
        )


def parse_parameter_assignments(values: Sequence[str]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"parameter must use ID=VALUE: {item}")
        parameter_id, raw_value = item.split("=", 1)
        if not parameter_id:
            raise ValueError(f"parameter ID is empty: {item}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"parameter value is not a number: {item}") from exc
        parameters.append({"Id": parameter_id, "Value": value})
    if not parameters:
        raise ValueError("at least one parameter is required")
    return parameters


def build_named_operation(command: str, args: Mapping[str, Any] | None = None) -> APIOperation:
    values = dict(args or {})
    model_uid = values.get("model_uid")
    needs_current = bool(values.get("use_current_model", model_uid is None))

    if command == "cubism_api.register":
        return APIOperation(command, "RegisterPlugin", "1.0.0", requires_approval=False)
    if command == "cubism_api.get_approval":
        return APIOperation(command, "GetIsApproval", "0.9.0", requires_approval=False)
    if command == "cubism_api.get_documents":
        return APIOperation(command, "GetDocuments", "0.9.1", {})
    if command == "cubism_api.get_document_snapshot":
        return APIOperation(command, "GetDocumentSnapshot", "", {})
    if command == "cubism_api.get_current_model_uid":
        return APIOperation(command, "GetCurrentModelUID", "0.9.1", {})
    if command == "cubism_api.get_current_edit_mode":
        return APIOperation(command, "GetCurrentEditMode", "0.9.0", {})
    if command == "cubism_api.get_parameters":
        parameter_data: dict[str, Any] = {"ModelUID": str(model_uid)} if model_uid else {}
        return APIOperation(
            command,
            "GetParameters",
            "1.0.1",
            parameter_data,
            needs_current_model=needs_current,
        )
    if command == "cubism_api.get_parameter_values":
        value_data: dict[str, Any] = {"ModelUID": str(model_uid)} if model_uid else {}
        ids = values.get("ids")
        if ids is not None and not isinstance(ids, list):
            raise ValueError("get_parameter_values ids must be a list")
        if isinstance(ids, list) and ids:
            value_data["Ids"] = [str(value) for value in ids]
        return APIOperation(
            command,
            "GetParameterValues",
            "0.9.1",
            value_data,
            needs_current_model=needs_current,
        )
    if command == "cubism_api.set_parameter_values":
        parameters = values.get("parameters")
        if not isinstance(parameters, list) or not parameters:
            raise ValueError("set_parameter_values requires a non-empty parameters list")
        set_data: dict[str, Any] = {"Parameters": parameters}
        if model_uid:
            set_data["ModelUID"] = str(model_uid)
        return APIOperation(
            command,
            "SetParameterValues",
            "0.9.1",
            set_data,
            needs_current_model=needs_current,
        )
    if command == "cubism_api.clear_parameter_values":
        clear_data = {"ModelUID": str(model_uid)} if model_uid else {}
        return APIOperation(
            command,
            "ClearParameterValues",
            "0.9.1",
            clear_data,
            needs_current_model=needs_current,
        )
    if command == "cubism_api.send_log":
        message = values.get("message")
        if not isinstance(message, str) or not message:
            raise ValueError("send_log requires a message")
        if len(message) > 5000:
            raise ValueError("send_log message must not exceed 5000 characters")
        log_type = str(values.get("type", "info"))
        if log_type not in {"info", "warning"}:
            raise ValueError("send_log type must be info or warning")
        return APIOperation(
            command,
            "SendCubismLog",
            "0.9.3",
            {"Type": log_type, "Message": message, "Display": bool(values.get("display", True))},
        )
    raise ValueError(f"unknown named Cubism API command: {command}")


def _resolve_model_uid(operation: APIOperation, model_uid: str) -> APIOperation:
    data = dict(operation.data)
    data["ModelUID"] = model_uid
    return replace(operation, data=data, needs_current_model=False)


def _redact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    if "Token" in redacted:
        redacted["Token"] = "<redacted>"
    data = redacted.get("Data")
    if isinstance(data, Mapping) and "Token" in data:
        redacted["Data"] = {**data, "Token": "<redacted>"}
    return redacted


def plan_operation(operation: APIOperation, options: ConnectionOptions) -> dict[str, Any]:
    stored_token = options.token or TokenStore(options.token_file).load()
    register_data: dict[str, Any] = {"Name": options.app_name}
    if stored_token:
        register_data["Token"] = stored_token
    sequence = [build_request("RegisterPlugin", register_data, request_id="register")]
    placeholder_token = stored_token or "<token-from-RegisterPlugin>"

    if operation.command != "cubism_api.register":
        sequence.append(
            build_request(
                "GetIsApproval",
                token=placeholder_token,
                request_id="approval",
            )
        )
        planned_operation = operation
        if operation.needs_current_model:
            sequence.append(
                build_request(
                    "GetCurrentModelUID",
                    token=placeholder_token,
                    request_id="current-model",
                )
            )
            planned_operation = _resolve_model_uid(operation, "<current-model-uid>")
        if operation.command == "cubism_api.get_document_snapshot":
            sequence.extend(
                [
                    build_request(
                        "GetDocuments",
                        token=placeholder_token,
                        request_id="documents",
                    ),
                    build_request(
                        "GetCurrentModelUID",
                        token=placeholder_token,
                        request_id="current-model",
                    ),
                ]
            )
        elif operation.command != "cubism_api.get_approval":
            sequence.append(
                build_request(
                    planned_operation.method,
                    planned_operation.data,
                    version=planned_operation.version,
                    token=placeholder_token,
                    request_id="command",
                )
            )

    return {
        "status": "planned",
        "mode": "dry-run",
        "url": options.url,
        "command": operation.command,
        "requests": [_redact_payload(payload) for payload in sequence],
    }


async def execute_operation(
    operation: APIOperation,
    options: ConnectionOptions,
    *,
    transport: RequestTransport | None = None,
) -> dict[str, Any]:
    async with CubismAPISession(options, transport=transport) as session:
        return await session.run(operation)


class CubismAPISession:
    """Keep one registered WebSocket connection across related API operations."""

    def __init__(
        self,
        options: ConnectionOptions,
        *,
        transport: RequestTransport | None = None,
    ) -> None:
        self.options = options
        self._provided_transport = transport
        self._owned_transport: WebSocketTransport | None = None
        self.client: CubismClient | None = None
        self.approved: bool | None = None

    async def __aenter__(self) -> CubismAPISession:
        active_transport = self._provided_transport
        if active_transport is None:
            self._owned_transport = WebSocketTransport(
                self.options.url,
                timeout=self.options.timeout,
            )
            active_transport = await self._owned_transport.__aenter__()

        try:
            store = TokenStore(self.options.token_file)
            token = self.options.token or store.load()
            self.client = CubismClient(active_transport, token=token)
            registered_token = await self.client.register(self.options.app_name)
            store.save(registered_token)
            return self
        except BaseException as exc:
            if self._owned_transport is not None:
                await self._owned_transport.__aexit__(type(exc), exc, exc.__traceback__)
                self._owned_transport = None
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._owned_transport is not None:
            await self._owned_transport.__aexit__(exc_type, exc, traceback)

    async def run(self, operation: APIOperation) -> dict[str, Any]:
        if self.client is None:
            raise CubismAPIError("Cubism API session is not connected")
        if operation.command == "cubism_api.register":
            return {
                "status": "completed",
                "command": operation.command,
                "registered": True,
                "token_saved": str(self.options.token_file.resolve()),
            }

        if operation.command == "cubism_api.get_approval":
            self.approved = await self.client.get_approval()
            return {
                "status": "completed",
                "command": operation.command,
                "approved": self.approved,
            }
        if self.approved is None:
            self.approved = await self.client.get_approval()
        if operation.requires_approval and not self.approved:
            raise CubismApprovalRequired(
                "Cubism External Application IntegrationでAllowを有効にしてください"
            )

        if operation.command == "cubism_api.get_document_snapshot":
            documents = await self.client.call(
                APIOperation("cubism_api.get_documents", "GetDocuments", "0.9.1")
            )
            current_model = await self.client.call(
                APIOperation(
                    "cubism_api.get_current_model_uid",
                    "GetCurrentModelUID",
                    "0.9.1",
                )
            )
            return {
                "status": "completed",
                "command": operation.command,
                "response": {
                    "Documents": documents.get("Data", {}),
                    "CurrentModel": current_model.get("Data", {}),
                },
            }

        resolved = operation
        if operation.needs_current_model:
            uid_response = await self.client.call(
                APIOperation(
                    "cubism_api.get_current_model_uid",
                    "GetCurrentModelUID",
                    "0.9.1",
                )
            )
            uid_data = uid_response.get("Data")
            model_uid = uid_data.get("ModelUID") if isinstance(uid_data, Mapping) else None
            if not model_uid:
                raise CubismAPIError("GetCurrentModelUID did not return ModelUID")
            resolved = _resolve_model_uid(operation, str(model_uid))

        response = await self.client.call(resolved)
        return {
            "status": "completed",
            "command": operation.command,
            "response": response,
        }

def run_named_api_command(
    command: str,
    args: Mapping[str, Any] | None = None,
    *,
    execute: bool = False,
    options: ConnectionOptions | None = None,
    transport: RequestTransport | None = None,
) -> dict[str, Any]:
    operation = build_named_operation(command, args)
    active_options = options or ConnectionOptions()
    if not execute:
        return plan_operation(operation, active_options)
    return asyncio.run(execute_operation(operation, active_options, transport=transport))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live2D Cubism External API client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--token")
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="connect to Cubism; omitted means dry-run",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "register",
        "get-approval",
        "get-documents",
        "get-document-snapshot",
        "get-current-model-uid",
        "get-current-edit-mode",
    ):
        sub.add_parser(name)

    get_parameters = sub.add_parser("get-parameters")
    get_parameters.add_argument("--model-uid")

    get_values = sub.add_parser("get-parameter-values")
    get_values.add_argument("--model-uid")
    get_values.add_argument("--id", action="append", dest="ids")

    set_values = sub.add_parser("set-parameter-values")
    set_values.add_argument("--model-uid")
    set_values.add_argument("--param", action="append", required=True)

    clear_values = sub.add_parser("clear-parameter-values")
    clear_values.add_argument("--model-uid")

    send_log = sub.add_parser("send-log")
    send_log.add_argument("message")
    send_log.add_argument("--type", choices=("info", "warning"), default="info")
    send_log.add_argument("--no-display", action="store_true")
    return parser


def _command_from_cli(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    command = f"cubism_api.{args.command.replace('-', '_')}"
    values: dict[str, Any] = {}
    if hasattr(args, "model_uid"):
        values["model_uid"] = args.model_uid
        values["use_current_model"] = args.model_uid is None
    if args.command == "get-parameter-values" and args.ids:
        values["ids"] = args.ids
    if args.command == "set-parameter-values":
        values["parameters"] = parse_parameter_assignments(args.param)
    if args.command == "send-log":
        values.update(
            {"message": args.message, "type": args.type, "display": not args.no_display}
        )
    return command, values


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = ConnectionOptions(
        host=args.host,
        port=args.port,
        app_name=args.app_name,
        token=args.token,
        token_file=args.token_file,
        timeout=args.timeout,
    )
    try:
        command, values = _command_from_cli(args)
        result = run_named_api_command(command, values, execute=args.execute, options=options)
    except (ValueError, CubismAPIError, OSError, TimeoutError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
