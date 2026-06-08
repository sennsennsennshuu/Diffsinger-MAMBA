"""
Cloud (AutoDL) helper for the DiffSinger-SSM project.

Usage (from project root):
    python tools/cloud.py exec "<remote-shell-command>"
    python tools/cloud.py get <remote-path> [<local-path>]
    python tools/cloud.py put <local-path> <remote-path>
    python tools/cloud.py mirror-up   <local-rel-path> <remote-rel-path>
    python tools/cloud.py mirror-down <remote-rel-path> <local-rel-path>

Connection settings come from `tools/cloud_config.json` next to this file.
The config holds host/port/user and password (locally only).
"""

from __future__ import annotations

import json
import os
import posixpath
import sys
from pathlib import Path

import paramiko


_HERE = Path(__file__).resolve().parent
_CFG = _HERE / "cloud_config.json"


def _load_cfg() -> dict:
    if not _CFG.exists():
        raise FileNotFoundError(f"Missing cloud config: {_CFG}")
    with _CFG.open("r", encoding="utf-8") as f:
        return json.load(f)


def _client() -> paramiko.SSHClient:
    cfg = _load_cfg()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=cfg["host"],
        port=int(cfg.get("port", 22)),
        username=cfg.get("user", "root"),
        password=cfg.get("password"),
        key_filename=cfg.get("key_filename"),
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return cli


def cmd_exec(argv: list[str]) -> int:
    if not argv:
        print("usage: cloud.py exec <command>", file=sys.stderr)
        return 2
    command = argv[0]
    cli = _client()
    try:
        stdin, stdout, stderr = cli.exec_command(command, timeout=None)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            sys.stdout.write(out)
            if not out.endswith("\n"):
                sys.stdout.write("\n")
        if err:
            sys.stderr.write(err)
            if not err.endswith("\n"):
                sys.stderr.write("\n")
        return rc
    finally:
        cli.close()


def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    parts = remote_path.split("/")
    cur = ""
    for p in parts:
        if not p:
            cur = "/" if not cur else cur
            continue
        cur = posixpath.join(cur, p) if cur else "/" + p if remote_path.startswith("/") else p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def cmd_get(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: cloud.py get <remote-path> [<local-path>]", file=sys.stderr)
        return 2
    remote = argv[0]
    local = argv[1] if len(argv) >= 2 else os.path.basename(remote)
    cli = _client()
    try:
        sftp = cli.open_sftp()
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, local)
        sftp.close()
        print(f"GET {remote} -> {local}")
        return 0
    finally:
        cli.close()


def cmd_put(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cloud.py put <local-path> <remote-path>", file=sys.stderr)
        return 2
    local = argv[0]
    remote = argv[1]
    cli = _client()
    try:
        sftp = cli.open_sftp()
        _ensure_remote_dir(sftp, posixpath.dirname(remote))
        sftp.put(local, remote)
        sftp.close()
        print(f"PUT {local} -> {remote}")
        return 0
    finally:
        cli.close()


def _walk_local(root: Path):
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            yield Path(dp) / fn


def cmd_mirror_up(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cloud.py mirror-up <local-dir> <remote-dir>", file=sys.stderr)
        return 2
    local_dir = Path(argv[0]).resolve()
    remote_dir = argv[1].rstrip("/")
    if not local_dir.is_dir():
        print(f"local dir not found: {local_dir}", file=sys.stderr)
        return 2
    cli = _client()
    try:
        sftp = cli.open_sftp()
        _ensure_remote_dir(sftp, remote_dir)
        for f in _walk_local(local_dir):
            rel = f.relative_to(local_dir).as_posix()
            target = posixpath.join(remote_dir, rel)
            _ensure_remote_dir(sftp, posixpath.dirname(target))
            sftp.put(str(f), target)
            print(f"PUT {rel}")
        sftp.close()
        return 0
    finally:
        cli.close()


def cmd_mirror_down(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cloud.py mirror-down <remote-dir> <local-dir>", file=sys.stderr)
        return 2
    remote_dir = argv[0].rstrip("/")
    local_dir = Path(argv[1]).resolve()
    cli = _client()
    try:
        sftp = cli.open_sftp()

        def _walk_remote(rd: str):
            for entry in sftp.listdir_attr(rd):
                full = posixpath.join(rd, entry.filename)
                if (entry.st_mode & 0o170000) == 0o040000:
                    yield from _walk_remote(full)
                else:
                    yield full

        local_dir.mkdir(parents=True, exist_ok=True)
        for full in _walk_remote(remote_dir):
            rel = posixpath.relpath(full, remote_dir)
            target = local_dir / rel.replace("/", os.sep)
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(full, str(target))
            print(f"GET {rel}")
        sftp.close()
        return 0
    finally:
        cli.close()


_DISPATCH = {
    "exec": cmd_exec,
    "get": cmd_get,
    "put": cmd_put,
    "mirror-up": cmd_mirror_up,
    "mirror-down": cmd_mirror_down,
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    op = sys.argv[1]
    fn = _DISPATCH.get(op)
    if not fn:
        print(f"unknown op: {op}", file=sys.stderr)
        return 2
    return fn(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())