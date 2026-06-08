"""Pull aco + var artifacts from the cloud `artifacts_cloud/<exp>/` directories.

Usage:
    python tools/pull_artifacts.py [exp_name ...]

Defaults to pulling both `var_testssm2` and `aco_testssm2`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import cloud  # noqa: E402

REMOTE_BASE = "/root/autodl-tmp/Diffsinger-main-SSM/artifacts_cloud"
LOCAL_BASE = ROOT / "artifacts_cloud"


def _pull(remote_path: str, local_path: Path, max_retries: int = 4) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, max_retries + 1):
        cli = cloud._client()
        try:
            sftp = cli.open_sftp()
            t0 = time.time()
            sftp.get(remote_path, str(local_path))
            sftp.close()
            size = os.path.getsize(local_path)
            print(f"OK   {local_path.name}  {size/1e6:.2f} MB  in {time.time()-t0:.1f}s")
            return
        except Exception as e:
            last_err = e
            print(f"FAIL {local_path.name} (attempt {attempt}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
        finally:
            cli.close()
    raise RuntimeError(f"giving up on {remote_path}: {last_err}")


def _list_remote(exp: str) -> list[str]:
    cli = cloud._client()
    try:
        sftp = cli.open_sftp()
        names = sftp.listdir(f"{REMOTE_BASE}/{exp}")
        sftp.close()
        return names
    finally:
        cli.close()


def main() -> int:
    exps = sys.argv[1:] or ["var_testssm2", "aco_testssm2"]
    for exp in exps:
        print(f"\n==== pulling {exp} ====")
        names = _list_remote(exp)
        for n in names:
            _pull(f"{REMOTE_BASE}/{exp}/{n}", LOCAL_BASE / exp / n)
    print("\nAll files mirrored to", LOCAL_BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())