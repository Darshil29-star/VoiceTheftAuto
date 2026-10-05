"""Download model weights listed in models/registry.yaml into CHECKPOINT_DIR.

Explicit only — the application never downloads weights at startup unless
ALLOW_REMOTE_MODEL_DOWNLOAD=true.

    python scripts/download_models.py --list
    python scripts/download_models.py --only florence2-base grounding-dino-tiny slimsam-50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.models.registry import get_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="model ids to download (default: all enabled ML models)")
    parser.add_argument("--list", action="store_true", help="show models and whether weights are present")
    args = parser.parse_args()

    settings = get_settings()
    registry = get_registry()
    ml = {mid: s for mid, s in registry.models.items() if s.kind == "ml"}

    if args.list:
        for mid, spec in ml.items():
            present = spec.weights_present(settings.checkpoint_dir)
            print(f"{mid:22s} {spec.size_mb or 0:>6.0f} MB  {spec.license or '?':12s} {'present' if present else 'missing'}")
        return 0

    from huggingface_hub import snapshot_download

    targets = args.only or [m for m, s in ml.items() if s.enabled]
    unknown = [t for t in targets if t not in ml]
    if unknown:
        print(f"Unknown model ids: {unknown}. Known: {list(ml)}")
        return 2

    for mid in targets:
        spec = ml[mid]
        dest = spec.path(settings.checkpoint_dir)
        print(f"==> {mid}: {spec.repo_id} -> {dest} (~{spec.size_mb} MB, license {spec.license})", flush=True)
        snapshot_download(repo_id=spec.repo_id, local_dir=dest, allow_patterns=spec.allow_patterns)
        print(f"    done: {mid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
