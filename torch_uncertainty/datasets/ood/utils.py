import logging
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _safe_extract(tar: tarfile.TarFile, path: Path):
    """Safely extract tar members into `path`, preventing path traversal."""
    base = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(base) + os.sep):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extract(member, path)


def _safe_extract_zip(zf: zipfile.ZipFile, path: Path):
    """Safely extract zip members into `path`, preventing path traversal."""
    base = path.resolve()
    for member in zf.namelist():
        member_path = (path / member).resolve()
        if not str(member_path).startswith(str(base) + os.sep):
            raise RuntimeError(f"Unsafe path in zip archive: {member}")
        zf.extract(member, path)


class FileListDataset(Dataset):
    def __init__(self, root: str | Path, list_file: str | Path, name=None, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.dataset_name = name
        self.samples = []
        with Path(list_file).open() as f:
            for line in f:
                path_str, lbl_str = line.strip().rsplit(maxsplit=1)
                self.samples.append((self.root / path_str, int(lbl_str)))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return the (image, label) sample at index `idx`."""
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


OOD_SPLITS = {
    "CIFAR10": {
        "test": {
            "cifar10": "splits/cifar10/test_ood_cifar10.txt",
        },
        "val": {
            "tinyimagenet": "splits/cifar10/val_tin.txt",
        },
        "near": {
            "cifar100": "splits/cifar10/test_cifar100.txt",
            "tinyimagenet": "splits/cifar10/test_tin.txt",
        },
        "far": {
            "mnist": "splits/cifar10/test_mnist.txt",
            "svhn": "splits/cifar10/test_svhn.txt",
            "texture": "splits/cifar10/test_texture.txt",
            "places365": "splits/cifar10/test_places365.txt",
        },
    },
    "CIFAR100": {
        "test": {
            "cifar100": "splits/cifar100/test_ood_cifar100.txt",
        },
        "val": {
            "tinyimagenet": "splits/cifar100/val_tin.txt",
        },
        "near": {
            "cifar10": "splits/cifar100/test_cifar10.txt",
            "tinyimagenet": "splits/cifar100/test_tin.txt",
        },
        "far": {
            "mnist": "splits/cifar100/test_mnist.txt",
            "svhn": "splits/cifar100/test_svhn.txt",
            "texture": "splits/cifar100/test_texture.txt",
            "places365": "splits/cifar100/test_places365.txt",
        },
    },
    "imagenet200": {
        "test": {
            "imagenet1k": "splits/imagenet200/test_ood_imagenet200.txt",
        },
        "val": {
            "openimage_o": "splits/imagenet200/val_openimage_o.txt",
        },
        "near": {
            "ssb_hard": "splits/imagenet200/test_ssb_hard.txt",
            "ninco": "splits/imagenet200/test_ninco.txt",
        },
        "far": {
            "inaturalist": "splits/imagenet200/test_inaturalist.txt",
            "texture": "splits/imagenet200/test_textures.txt",
            "openimage_o": "splits/imagenet200/test_openimage_o.txt",
        },
    },
    "imagenet1k": {
        "test": {
            "imagenet1k": "splits/imagenet1k/test_ood_imagenet.txt",
        },
        "val": {
            "openimage_o": "splits/imagenet1k/val_openimage_o.txt",
        },
        "near": {
            "ssb_hard": "splits/imagenet1k/test_ssb_hard.txt",
            "ninco": "splits/imagenet1k/test_ninco.txt",
        },
        "far": {
            "inaturalist": "splits/imagenet1k/test_inaturalist.txt",
            "texture": "splits/imagenet1k/test_textures.txt",
            "openimage_o": "splits/imagenet1k/test_openimage_o.txt",
        },
    },
}


ZENODO_INFO = {
    "ninco": {
        "url": "https://zenodo.org/record/8013288/files/NINCO_all.tar.gz",
        "filename": "NINCO_all.tar.gz",
        "extract_paths": ["NINCO"],
    },
    "openimage_o": {
        "url": "https://zenodo.org/records/10540831/files/OpenImage-O.zip",
        "filename": "OpenImage-O.zip",
        "extract_paths": ["openimage-o"],
    },
}

HF_REPO_INFO: dict[str, dict[str, str]] = {
    "cifar10": {
        "repo_id": "torch-uncertainty/Cifar10",
        "zip_filename": "cifar10.zip",
    },
    "cifar100": {
        "repo_id": "torch-uncertainty/Cifar100",
        "zip_filename": "cifar100.zip",
    },
    "mnist": {
        "repo_id": "torch-uncertainty/MNIST",
        "zip_filename": "mnist.zip",
    },
    "texture": {
        "repo_id": "torch-uncertainty/Texture",
        "zip_filename": "texture.zip",
    },
    "places365": {
        "repo_id": "torch-uncertainty/Places365",
        "zip_filename": "places365.zip",
    },
    "svhn": {
        "repo_id": "torch-uncertainty/SVHN",
        "zip_filename": "svhn.zip",
    },
    "tinyimagenet": {
        "repo_id": "torch-uncertainty/tiny-imagenet-200",
        "zip_filename": "tin.zip",
    },
    "ssb_hard": {
        "repo_id": "torch-uncertainty/SSB_hard",
        "zip_filename": "ssb_hard.zip",
    },
    "inaturalist": {
        "repo_id": "torch-uncertainty/inaturalist",
        "zip_filename": "inaturalist.zip",
    },
    "imagenet1k": {
        "repo_id": "torch-uncertainty/Imagenet1k",
        "zip_filename": "imagenet_1k.zip",
    },
}


def download_and_extract_hf_dataset(
    name: str,
    root: Path,
) -> Path:
    """- If name is 'ninco' or 'openimage_o', download from Zenodo and extract once.
    - Otherwise fall back to HF_REPO_INFO + hf_hub_download.
    Returns the path to the folder you should use as 'root' for FileListDataset.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def _download_zenodo():
        logger.info("📥 Downloading '%s' from Zenodo…", name)
        urllib.request.urlretrieve(info["url"], archive_path)  # noqa: S310

    def _attempt_extract_archive():
        logger.info("📂 Extracting '%s'…", archive_path.name)
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                _safe_extract_zip(zf, root)
        elif archive_path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(archive_path, "r:gz") as tf:
                _safe_extract(tf, root)
        else:
            raise RuntimeError(f"Unknown archive format: {archive_path}")

    if name in ZENODO_INFO:
        info = ZENODO_INFO[name]
        archive_path = root / info["filename"]

        for rel in info["extract_paths"]:
            candidate = root / rel
            if candidate.exists():
                return candidate

        if not archive_path.exists():
            _download_zenodo()

        try:
            _attempt_extract_archive()
        except (RuntimeError, zipfile.BadZipFile, tarfile.TarError, OSError) as e:
            logger.warning("Extraction failed (%s), re-downloading and retrying…", e)
            archive_path.unlink(missing_ok=True)
            _download_zenodo()
            _attempt_extract_archive()

        for rel in info["extract_paths"]:
            candidate = root / rel
            if candidate.exists():
                return candidate

        raise RuntimeError(
            f"Extraction succeeded but none of {info['extract_paths']} were found under {root!r}"
        )

    hf_info = HF_REPO_INFO.get(name)
    if hf_info is None:
        raise KeyError(f"No HF_REPO_INFO entry for {name}")

    repo_id = hf_info["repo_id"]
    zip_fname = hf_info["zip_filename"]
    target_dir = root / Path(zip_fname).stem

    if target_dir.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "📥 Downloading %r from HF Hub (%s/%s)…",
        name,
        repo_id,
        zip_fname,
    )
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_fname,
        repo_type="dataset",
    )

    def _extract_hf_zip():
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, target_dir)

    try:
        _extract_hf_zip()
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning("HF Hub zip extract failed (%s), re-downloading and retrying…", e)
        Path(zip_path).unlink(missing_ok=True)
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_fname,
            repo_type="dataset",
            force_download=True,
        )
        _extract_hf_zip()

    return target_dir


def get_ood_datasets(
    root: str | Path,
    dataset_id: str,
    transform=None,
) -> tuple[FileListDataset, dict[str, FileListDataset], dict[str, FileListDataset]]:
    """Ensure all OOD splits are downloaded and extracted via HF_REPO_INFO."""
    root = Path(root)
    splits_base = download_and_extract_splits_from_hf(root=Path(root))

    def _resolve_txt(rel_txt: str) -> Path:
        rel = rel_txt.lstrip("/")
        rel = rel.removeprefix("splits/")
        return splits_base / rel

    cfg = OOD_SPLITS.get(dataset_id)
    if cfg is None:
        raise KeyError(f"No OOD_SPLITS for {dataset_id}")

    def build(name: str, rel_txt: str):
        data_dir = download_and_extract_hf_dataset(name, root)
        txt = _resolve_txt(rel_txt)
        return FileListDataset(root=data_dir, list_file=txt, transform=transform, name=name)

    test_name, test_txt = next(iter(cfg["test"].items()))
    test_ood = build(test_name, test_txt)

    val_name, val_txt = next(iter(cfg["val"].items()))
    val_ood = build(val_name, val_txt)

    near_oods = {n: build(n, p) for n, p in cfg["near"].items()}
    far_oods = {n: build(n, p) for n, p in cfg["far"].items()}

    return test_ood, val_ood, near_oods, far_oods


def download_and_extract_splits_from_hf(
    root: str | Path,
    repo_id="torch-uncertainty/ood-datasets-splits",
    zip_filename="splits.zip",
) -> Path:
    """Download a zip that contains the 'splits/' tree from HF and extract it once.
    Returns the path to the extracted 'splits' directory (or the extracted root if it already is 'splits/').
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    target_dir = root / Path(zip_filename).stem  # e.g. <root>/splits

    def _is_valid_splits_dir(p: Path) -> bool:
        # valid if it has a 'splits/' subdir OR known subfolders OR any .txt files inside
        if (p / "splits").exists():
            return True
        for sub in ("cifar10", "cifar100", "imagenet1k", "imagenet200"):
            if (p / sub).exists():
                return True
        return any(p.rglob("*.txt"))

    # EARLY RETURN ONLY IF VALID
    if target_dir.exists() and _is_valid_splits_dir(target_dir):
        return (target_dir / "splits") if (target_dir / "splits").exists() else target_dir

    # (Re)create and fetch
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📥 Downloading splits from HF Hub (%s/%s)…", repo_id, zip_filename)
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_filename,
        repo_type="dataset",  # change to "model" if hosted as a model
    )

    def _extract_zip(zp: Path, out: Path):
        with zipfile.ZipFile(zp, "r") as zf:
            _safe_extract_zip(zf, out)

    try:
        _extract_zip(Path(zip_path), target_dir)
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning("Splits zip extract failed (%s), re-downloading and retrying…", e)
        Path(zip_path).unlink(missing_ok=True)
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type="dataset",
            force_download=True,
        )
        _extract_zip(Path(zip_path), target_dir)

    # Choose the actual splits dir to return
    final_dir = (target_dir / "splits") if (target_dir / "splits").exists() else target_dir

    # VALIDATE POST-EXTRACT
    if not _is_valid_splits_dir(final_dir):
        raise FileNotFoundError(
            f"No split files found under {final_dir}. "
            f"Check the structure of {repo_id}:{zip_filename}."
        )

    return final_dir
