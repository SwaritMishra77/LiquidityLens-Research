"""Import the original v0.2 SEC cache into a source checkout.

The published snapshot is a separate archive. Verify every checksum before
copying only data files; never execute or deserialize model files in the ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

PREFIX = "liquiditylens/"
DATA_PREFIX = PREFIX + "data/"
CHECKSUM_FILE = PREFIX + "CHECKSUMS.sha256"


def import_snapshot(archive: Path, destination: Path) -> dict[str, int]:
    archive = Path(archive)
    destination = Path(destination).resolve()
    with ZipFile(archive) as zf:
        members = {item.filename: item for item in zf.infolist() if not item.is_dir()}
        if len(members) != sum(not item.is_dir() for item in zf.infolist()):
            raise ValueError("Duplicate ZIP member names")
        if CHECKSUM_FILE not in members:
            raise ValueError("Missing checksum manifest")
        for item in members.values():
            parts = PurePosixPath(item.filename).parts
            if not item.filename.startswith(PREFIX) or ".." in parts or "\\" in item.filename:
                raise ValueError(f"Unsafe ZIP member: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink in ZIP: {item.filename}")

        expected = {}
        for line in zf.read(CHECKSUM_FILE).decode("utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            full_name = PREFIX + name
            if separator != "  " or len(digest) != 64 or full_name in expected:
                raise ValueError("Invalid checksum manifest")
            expected[full_name] = digest
        metadata = {CHECKSUM_FILE, PREFIX + "PACKAGE_MANIFEST.json"}
        if set(expected) != set(members) - metadata:
            raise ValueError("Checksum manifest does not cover exactly the ZIP members")

        # Check the complete original archive, including code and reports, before
        # importing data. This catches modified source or results as well as data.
        for name, digest in expected.items():
            with zf.open(name) as source:
                hasher = hashlib.sha256()
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(block)
                if hasher.hexdigest() != digest:
                    raise ValueError(f"Checksum mismatch: {name}")

        selected = [name for name in expected if name.startswith(DATA_PREFIX)]
        if not selected:
            raise ValueError("Archive has no data files")
        total = 0
        for name in selected:
            relative = PurePosixPath(name.removeprefix(PREFIX))
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".importing")
            with zf.open(name) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.replace(temporary, target)
            total += target.stat().st_size
    return {"files": len(selected), "bytes": total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to LiquidityLens_v0_2.zip")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(import_snapshot(args.archive, args.root))
