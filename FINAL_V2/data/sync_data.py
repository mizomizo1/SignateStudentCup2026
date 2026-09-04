from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

EXPECTED = {
    "train.csv": "7e7ef33cca57f8e1564004267cbcde31181ee947",
    "test.csv": "d6a352d79e211f3d2d1c0bd9c636d4a7af515474",
}


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def main() -> None:
    source_dir = ROOT / "data"
    for name, expected in EXPECTED.items():
        src = source_dir / name
        dst = HERE / name
        if not src.exists():
            raise FileNotFoundError(src)
        actual = git_blob_sha(src)
        if actual != expected:
            raise RuntimeError(f"unexpected source blob for {name}: {actual} != {expected}")
        shutil.copyfile(src, dst)
        copied = git_blob_sha(dst)
        if copied != expected:
            raise RuntimeError(f"copy verification failed for {name}: {copied} != {expected}")
        print(f"{name}: {expected}")


if __name__ == "__main__":
    main()
