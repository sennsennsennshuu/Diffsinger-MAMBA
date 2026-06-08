"""Robust per-file pull from the cloud var_testssm2 export.

`tools/cloud.py mirror-down` opens one long-lived SFTP session and gets
all files; that connection occasionally drops half-way through (~40 s in)
on the autodl link, leaving the mirror partial.  This script gets each
file in its own SFTP session with retries, so a transient drop costs us
the file-in-flight at most.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paramiko  # noqa: E402

from tools import cloud  # noqa: E402


REMOTE_DIR = "/root/autodl-tmp/Diffsinger-main-SSM/onnx_out_var_testssm2"
LOCAL_DIR  = ROOT / "artifacts_cloud" / "var_testssm2"

FILES = [
    "dictionary-ja.txt",
    "dictionary-zh.txt",
    "dsconfig.yaml",
    "var_testssm2.dur.onnx",
    "var_testssm2.itako.emb",
    "var_testssm2.karasu.emb",
    "var_testssm2.kiritan.emb",
    "var_testssm2.languages.json",
    "var_testssm2.linguistic.onnx",
    "var_testssm2.opencpop.emb",
    "var_testssm2.phonemes.json",
    "var_testssm2.pitch.onnx",
    "var_testssm2.variance.onnx",
]


def _pull(name: str, max_retries: int = 4) -> None:
    remote = f"{REMOTE_DIR}/{name}"
    local  = LOCAL_DIR / name
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, max_retries + 1):
        cli = cloud._client()
        try:
            sftp = cli.open_sftp()
            t0 = time.time()
            sftp.get(remote, str(local))
            sftp.close()
            size = os.path.getsize(local)
            print(f"OK   {name}  {size/1e6:.2f} MB  in {time.time()-t0:.1f}s")
            return
        except Exception as e:
            last_err = e
            print(f"FAIL {name} (attempt {attempt}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
        finally:
            cli.close()
    raise RuntimeError(f"giving up on {name}: {last_err}")


def main() -> int:
    for f in FILES:
        _pull(f)
    print("\nAll files mirrored to", LOCAL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())