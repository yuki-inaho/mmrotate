"""Shared helpers for cu12 GPU smoke scripts.

Each smoke script wraps a single callable that exercises one representative
CUDA op and either returns a dict ``{"cuda_op_device": str, ...}`` on success or
raises. This helper records that result with torch / cuda / device metadata.

Strict policy: ``success`` is true only when ``cuda_op_device == 'cuda'`` AND
``torch.cuda.is_available()`` is true. Mirrors the validated logic from the
``mmopenlab_cu12_sandbox`` reference work.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def collect_torch_info() -> dict[str, Any]:
    import torch

    cap = None
    name = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        c = torch.cuda.get_device_capability(0)
        cap = f"{c[0]}.{c[1]}"
        name = torch.cuda.get_device_name(0)
    return {
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "is_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "cuda_device_name": name,
        "compute_capability": cap,
    }


def module_version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, "__version__", None)


def run_smoke(
    *,
    module_name: str,
    runner: Callable[[], dict[str, Any]],
    output_json: Path | None,
    extra_meta: dict[str, Any] | None = None,
) -> int:
    """Run ``runner`` and (optionally) persist a JSON evidence file.

    Returns process exit code (0 on success, 2 on failure).
    """
    started = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "module": module_name,
        "module_version": module_version(module_name),
        "started_utc": started,
        "extra_meta": extra_meta or {},
    }
    try:
        payload["torch"] = collect_torch_info()
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        result = runner()
        if not isinstance(result, dict):
            raise TypeError(f"runner must return dict, got {type(result).__name__}")
        payload["op_result"] = result
        payload["error"] = None
    except Exception as exc:  # noqa: BLE001
        payload["op_result"] = None
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=15).splitlines(),
        }

    op_device = (payload["op_result"] or {}).get("cuda_op_device") if payload["op_result"] else None
    torch_ok = (payload["torch"] or {}).get("is_available") is True
    payload["cuda_op_device"] = op_device
    payload["success"] = (op_device == "cuda") and torch_ok and payload["error"] is None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    summary = {
        "module": module_name,
        "module_version": payload["module_version"],
        "success": payload["success"],
        "cuda_op_device": op_device,
        "error_type": (payload["error"] or {}).get("type") if payload["error"] else None,
    }
    if payload["error"]:
        summary["error_message"] = payload["error"]["message"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if payload["success"] else 2


def standard_argv(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path to write the full JSON evidence file",
    )
    return parser.parse_args()
