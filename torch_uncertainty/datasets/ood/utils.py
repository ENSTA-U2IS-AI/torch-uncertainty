from pathlib import Path
import zipfile
from typing import Dict, Tuple

from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from PIL import Image

class FileListDataset(Dataset):
    def __init__(self,
                 root: str | Path,
                 list_file: str | Path,
                 name=None,
                 transform=None):
        self.root = Path(root)
        self.transform = transform
        self.dataset_name=name
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                rel_path, lbl = line.strip().split()
                self.samples.append((self.root / rel_path, int(lbl)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label



OOD_SPLITS = {
    'CIFAR10': {
        'val': {
            'tinyimagenet': 'splits/cifar10/val_tin.txt',
        },
        'near': {
            'cifar100':     'splits/cifar10/test_cifar100.txt',
            'tinyimagenet': 'splits/cifar10/test_tin.txt',
        },
        'far': {
            'mnist':        'splits/cifar10/test_mnist.txt',
            'svhn':         'splits/cifar10/test_svhn.txt',
            'texture':      'splits/cifar10/test_texture.txt',
            'places365':    'splits/cifar10/test_places365.txt',
        },
    },
}



HF_REPO_INFO: Dict[str, Dict[str, str]] = {
    'cifar100': {
        'repo_id':     'torch-uncertainty/Cifar100',
        'zip_filename':'cifar100.zip',
    },
    'mnist': {
        'repo_id':     'torch-uncertainty/MNIST',
        'zip_filename':'mnist.zip',
    },    
    'texture': {
        'repo_id':     'torch-uncertainty/Texture',
        'zip_filename':'texture.zip',
    },
    
    'places365': {
        'repo_id':     'torch-uncertainty/Places365',
        'zip_filename':'places365.zip',
    },  

    'svhn': {
        'repo_id':     'torch-uncertainty/SVHN',
        'zip_filename':'svhn.zip',
    }, 

    'tinyimagenet': {
        'repo_id':     'torch-uncertainty/tiny-imagenet-200',
        'zip_filename':'tin.zip',
    },  


}


def download_and_extract_hf_dataset(
    name: str,
    root: Path,
) -> Path:
    """
    - Looks up HF_REPO_INFO[name] to get repo_id + zip_filename
    - Downloads that ZIP from HF Hub, then extracts it under
      root / zip_stem / … and returns that subfolder path.
    """
    info = HF_REPO_INFO.get(name)
    if info is None:
        raise KeyError(f"No HF_REPO_INFO entry for {name}")
    repo_id      = info['repo_id']
    zip_fname    = info['zip_filename']

    target_dir = root / Path(zip_fname).stem

    if target_dir.exists():
        return target_dir
    
    target_dir.mkdir(parents=True, exist_ok=True)


    print(f"📥 Downloading {name!r} from '{info['repo_id']}/{zip_fname}'...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_fname,
        repo_type="dataset",
    )
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir



def get_ood_datasets(
    root: str | Path,
    dataset_id: str,
    transform=None,
) -> Tuple[FileListDataset, Dict[str,FileListDataset], Dict[str,FileListDataset]]:
    """
    - Ensures all OOD splits are downloaded+extracted via HF_REPO_INFO
    - Returns (val_ood, near_oods, far_oods) as FileListDatasets
      using your splits under splits/…
    """
    root = Path(root)
    splits_base = Path(__file__).parent
    cfg = OOD_SPLITS.get(dataset_id)
    if cfg is None:
        raise KeyError(f"No OOD_SPLITS for {dataset_id}")

    def build(name: str, rel_txt: str):
        data_dir = download_and_extract_hf_dataset(name, root)
        txt = splits_base / rel_txt
        return FileListDataset(root=data_dir, list_file=txt, transform=transform,name=name)

    val_name, val_txt = next(iter(cfg['val'].items()))
    val_ood = build(val_name, val_txt)

    near_oods = {n: build(n, p) for n, p in cfg['near'].items()}
    far_oods  = {n: build(n, p) for n, p in cfg['far'].items()}

    return val_ood, near_oods, far_oods