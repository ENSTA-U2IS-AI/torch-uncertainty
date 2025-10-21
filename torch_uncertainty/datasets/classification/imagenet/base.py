import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)


class ImageNetVariation(ImageFolder):
    url: str | list[str]
    filename: str | list[str]
    tgz_md5: str | list[str]
    dataset_name: str
    root_appendix: str

    wnid_to_idx_url = (
        "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/"
        "classification/imagenet/classes.json"
    )
    wnid_to_idx_md5 = "1bcf467b49f735dbeb745249eae6b133"

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it."
            )

        super().__init__(
            root=self.root / Path(self.dataset_name),
            transform=transform,
            target_transform=target_transform,
        )
        self._repair_dataset()

    # ---------------- helpers ----------------

    def _md5_of(self, path: Path, chunk_size: int = 1 << 22) -> str:
        h = hashlib.md5()  # noqa: S324
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _check_integrity(self) -> bool:
        data_root = self.root / self.dataset_name
        if data_root.is_dir():
            logging.info("[integrity] extracted dir present: %s -> OK", data_root)
            return True

        if isinstance(self.filename, str):
            p = (self.root / Path(self.filename)).resolve()
            if p.exists():
                actual = self._md5_of(p)
                logging.info(
                    "[integrity] %s | MD5 got: %s | expected: %s | %s",
                    p,
                    actual,
                    self.tgz_md5,
                    "OK" if actual == self.tgz_md5 else "FAIL",
                )
            else:
                logging.info("[integrity] missing archive: %s", p)
            return check_integrity(p, self.tgz_md5)

        if isinstance(self.filename, list):
            ok = True
            for filename, md5 in zip(self.filename, self.tgz_md5, strict=True):
                p_root = (self.root / filename).resolve()
                if p_root.exists():
                    actual = self._md5_of(p_root)
                    logging.info(
                        "[integrity] %s | MD5 got: %s | expected: %s | %s",
                        p_root,
                        actual,
                        md5,
                        "OK" if actual == md5 else "FAIL",
                    )
                else:
                    logging.info("[integrity] missing archive at <root>/: %s", p_root)
                ok *= check_integrity(p_root, md5)
            return bool(ok)

        raise ValueError("filename must be str or list")

    def download(self) -> None:
        data_root = self.root / self.dataset_name
        if data_root.is_dir():
            logging.info("[download] extracted dir present -> skipping download/extract.")
            return

        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            if isinstance(self.filename, list):
                logging.info("[download] extracting existing valid archives...")
                for url, filename, md5 in zip(self.url, self.filename, self.tgz_md5, strict=True):
                    p_root = (self.root / filename).resolve()
                    if p_root.exists() and check_integrity(p_root, md5):
                        logging.info("[download] Extracting: %s", p_root)
                        download_and_extract_archive(
                            url,
                            self.root,
                            extract_root=self.root / self.root_appendix,
                            filename=filename,
                            md5=md5,
                        )
            return

        if isinstance(self.filename, str):
            p = (self.root / Path(self.filename)).resolve()
            if p.exists():
                actual = self._md5_of(p)
                logging.info(
                    "[download] existing %s | MD5 got: %s | expected: %s", p, actual, self.tgz_md5
                )
            download_and_extract_archive(
                self.url,
                self.root,
                extract_root=self.root / self.root_appendix,
                filename=self.filename,
                md5=self.tgz_md5,
            )
            return

        for url, filename, md5 in zip(self.url, self.filename, self.tgz_md5, strict=True):
            p_root = (self.root / filename).resolve()
            if p_root.exists():
                actual = self._md5_of(p_root)
                logging.info(
                    "[download] existing %s | MD5 got: %s | expected: %s", p_root, actual, md5
                )
            if not check_integrity(p_root, md5):
                logging.info("[download] fetching -> %s -> %s", url, p_root)
                download_and_extract_archive(
                    url,
                    self.root,
                    extract_root=self.root / self.root_appendix,
                    filename=filename,
                    md5=md5,
                )
            elif not data_root.is_dir():
                logging.info("[download] extracting valid existing: %s", p_root)
                download_and_extract_archive(
                    url,
                    self.root,
                    extract_root=self.root / self.root_appendix,
                    filename=filename,
                    md5=md5,
                )

    def _repair_dataset(self) -> None:
        path = self.root / "classes.json"
        if not check_integrity(path, self.wnid_to_idx_md5):
            download_url(self.wnid_to_idx_url, self.root, "classes.json", self.wnid_to_idx_md5)

        with (self.root / "classes.json").open() as file:
            self.wnid_to_idx = json.load(file)

        for i in range(len(self.samples)):
            wnid = Path(self.samples[i][0]).parts[-2]
            self.targets[i] = self.wnid_to_idx[wnid]
            self.samples[i] = (self.samples[i][0], self.targets[i])
