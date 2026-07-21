# mmrotate cu12 (torch 2.1.0 + cu121) task runner.
#
#   just sync        # provision .venv (cu12 deps) + put this mmrotate source on the path (.pth)
#   just smoke       # import mmrotate (mmcv/mmdet/mmengine guards) + box_iou_rotated on GPU
#   just env-doctor  # print python/torch/mmcv/mmdet/mmengine/mmrotate/gpu state

VENV := ".venv"
PY := justfile_directory() + "/" + VENV + "/bin/python"

default:
    @just --list

list:
    @just --list

# Provision the cu12 env (deps) and make this mmrotate source importable (.pth).
sync:
    uv sync
    # PyPI mmdet 3.3.0 ships mmcv_maximum_version='2.2.0' (blocks mmcv 2.2.0). Relax
    # the installed copy so the cu12 stack runs on mmcv 2.2.0 (matches our mmdet fork).
    sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
      "{{ VENV }}/lib/python3.10/site-packages/mmdet/__init__.py" 2>/dev/null || true
    printf '%s\n' "{{ justfile_directory() }}" > "{{ VENV }}/lib/python3.10/site-packages/_cu12_src.pth"
    @echo "Synced: cu12 deps + mmrotate source on path (.pth); installed mmdet guard relaxed for mmcv 2.2.0."

# Read-only environment triage (imports from /tmp to avoid source shadowing).
env-doctor:
    @echo "=== gpu ==="
    @nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    @echo ""
    @echo "=== python / packages ==="
    @if [ -x "{{ PY }}" ]; then \
        "{{ PY }}" --version; \
        for pkg in torch mmcv mmengine mmdet mmrotate numpy; do \
            (cd /tmp && "{{ PY }}" -c "import importlib; m=importlib.import_module('$pkg'); print('$pkg', getattr(m,'__version__','?'))") 2>/dev/null || echo "$pkg IMPORT_ERROR"; \
        done; \
        (cd /tmp && "{{ PY }}" -c "import torch; print('cuda_available', torch.cuda.is_available(), '| cuda', torch.version.cuda)"); \
    else \
        echo "{{ PY }} missing — run 'just sync' first."; \
    fi

# Run the mmrotate GPU smoke (import guards + box_iou_rotated on cuda).
smoke OUT="/tmp/smoke_mmrotate_cu12.json": sync
    "{{ PY }}" tools/cu12/smoke_mmrotate.py --output-json "{{ OUT }}"
