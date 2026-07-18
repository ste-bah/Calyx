#!/usr/bin/env python3
"""One fail-closed error envelope for every lawslice command."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import NoReturn


DEFAULT_CODE = "lawslice_error"
DEFAULT_REMEDIATION = (
    "inspect the typed context, correct the reported source or contract condition, "
    "and rerun without reusing a partial destination"
)


def _nonempty(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class StructuredError(RuntimeError):
    """A typed lawslice refusal with mandatory remediation and separate context."""

    code = DEFAULT_CODE
    default_remediation = DEFAULT_REMEDIATION

    def __init__(
        self,
        message: str,
        *,
        remediation: str | None = None,
        **context,
    ):
        nested_remediation = context.pop("remediation", None)
        self.context = context
        self.remediation = (
            _nonempty(remediation)
            or _nonempty(nested_remediation)
            or self.default_remediation
        )
        super().__init__(message)

    def record(self) -> dict:
        return {
            "code": _nonempty(self.code) or DEFAULT_CODE,
            "message": _nonempty(str(self)) or type(self).__name__,
            "remediation": self.remediation,
            "context": self.context,
        }


class StructuredArgumentParser(argparse.ArgumentParser):
    """Raise a typed error instead of printing argparse's unstructured refusal."""

    def _print_message(self, message, file=None) -> None:
        if not message:
            return
        stream = file or sys.stderr
        try:
            stream.write(message)
        except UnicodeEncodeError:
            encoding = getattr(stream, "encoding", None) or "ascii"
            escaped = message.encode(encoding, errors="backslashreplace").decode(
                encoding
            )
            stream.write(escaped)

    def error(self, message: str) -> NoReturn:
        raise ArgumentContractError(
            message,
            remediation="provide the required command and arguments exactly as shown in usage",
            usage=self.format_usage().strip(),
        )


class ArgumentContractError(StructuredError):
    code = "lawslice_argument_error"
    default_remediation = (
        "provide the required command and arguments exactly as shown in usage"
    )


def error_envelope(
    error: BaseException,
    *,
    code: str = DEFAULT_CODE,
    remediation: str = DEFAULT_REMEDIATION,
    context: dict | None = None,
) -> dict:
    """Normalize typed and untyped exceptions into the binding public envelope."""

    record = {}
    record_method = getattr(error, "record", None)
    if callable(record_method):
        candidate = record_method()
        if isinstance(candidate, dict):
            record = dict(candidate)

    error_context = record.get("context", getattr(error, "context", {}))
    if not isinstance(error_context, dict):
        error_context = {"invalid_context": repr(error_context)}
    else:
        error_context = dict(error_context)
    nested_remediation = error_context.pop("remediation", None)
    if context:
        error_context.update(context)

    return {
        "status": "error",
        "error_type": type(error).__name__,
        "code": (
            _nonempty(record.get("code"))
            or _nonempty(getattr(error, "code", None))
            or _nonempty(code)
            or DEFAULT_CODE
        ),
        "message": _nonempty(record.get("message"))
        or _nonempty(str(error))
        or type(error).__name__,
        "remediation": (
            _nonempty(record.get("remediation"))
            or _nonempty(getattr(error, "remediation", None))
            or _nonempty(nested_remediation)
            or _nonempty(remediation)
            or DEFAULT_REMEDIATION
        ),
        "context": error_context,
    }


def write_error(
    error: BaseException,
    *,
    code: str = DEFAULT_CODE,
    remediation: str = DEFAULT_REMEDIATION,
    context: dict | None = None,
    include_traceback: bool = True,
) -> dict:
    """Write one structured refusal before an optional diagnostic traceback."""

    record = error_envelope(
        error,
        code=code,
        remediation=remediation,
        context=context,
    )
    sys.stderr.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if include_traceback:
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    return record


def parse_cli_args(parser: argparse.ArgumentParser):
    """Preserve help exit 0; make every argument failure a structured exit 2."""

    try:
        return parser.parse_args()
    except SystemExit as error:
        if error.code == 0:
            raise
        write_error(
            error,
            code="lawslice_argument_error",
            remediation="provide the required command and arguments exactly as shown in usage",
            include_traceback=False,
        )
        raise
    except BaseException as error:
        write_error(
            error,
            code="lawslice_argument_error",
            remediation="provide the required command and arguments exactly as shown in usage",
            include_traceback=False,
        )
        raise SystemExit(2) from error
