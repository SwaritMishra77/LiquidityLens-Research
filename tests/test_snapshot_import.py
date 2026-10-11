import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from scripts.import_snapshot import import_snapshot


def make_archive(path, data=b"SEC cache", *, corrupt=False, extra=None):
    member = "liquiditylens/data/raw/facts/0001.json.gz"
    digest = hashlib.sha256(data).hexdigest()
    with ZipFile(path, "w") as archive:
        archive.writestr(member, data + (b"tampered" if corrupt else b""))
        if extra is not None:
            archive.writestr(extra, b"outside")
        checks = [f"{digest}  data/raw/facts/0001.json.gz"]
        if extra is not None:
            checks.append(f"{hashlib.sha256(b'outside').hexdigest()}  {extra.removeprefix('liquiditylens/')}")
        archive.writestr("liquiditylens/CHECKSUMS.sha256", "\n".join(checks))
        archive.writestr("liquiditylens/PACKAGE_MANIFEST.json", "{}")


class SnapshotImportTests(unittest.TestCase):
    def test_checks_and_extracts_data_only(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "snapshot.zip"
            make_archive(archive)
            result = import_snapshot(archive, root / "checkout")
            self.assertEqual(result, {"files": 1, "bytes": 9})
            self.assertEqual((root / "checkout/data/raw/facts/0001.json.gz").read_bytes(), b"SEC cache")

    def test_rejects_modified_data_before_extraction(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "snapshot.zip"
            make_archive(archive, corrupt=True)
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                import_snapshot(archive, root / "checkout")
            self.assertFalse((root / "checkout/data").exists())

    def test_rejects_path_traversal(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "snapshot.zip"
            make_archive(archive, extra="liquiditylens/../../escape.txt")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                import_snapshot(archive, root / "checkout")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
