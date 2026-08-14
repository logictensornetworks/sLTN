"""Download the datasets used in the sltn_notebooks examples.

Each ``download_xx_dataset`` function downloads the files for dataset ``xx``
into a target directory. If a file is already present it is left untouched and
the user is informed; otherwise it is downloaded and a confirmation is printed.
"""

from pathlib import Path
from urllib.request import urlretrieve

_RAW_BASE = "https://raw.githubusercontent.com/logictensornetworks/logictensornetworks/master/examples"

# Mapping of dataset name -> {local filename: remote url}
_DATASETS = {
    "iris": {
        "iris_training.csv": f"{_RAW_BASE}/multiclass_classification/iris_training.csv",
        "iris_test.csv": f"{_RAW_BASE}/multiclass_classification/iris_test.csv",
    },
    "crabs": {
        "crabs.dat": f"{_RAW_BASE}/multiclass_classification/crabs.dat",
    },
    "real-estate": {
        "real-estate.csv": f"{_RAW_BASE}/regression/real-estate.csv",
    },
}


def download_dataset(name: str, path):
    """Download every file of dataset ``name`` into ``path``.

    Files already present in ``path`` are skipped.
    """
    assert name in _DATASETS, (f"Unknown dataset '{name}'. Available "
                               f"datasets: {list(_DATASETS.keys())}")

    target_dir = Path(path)
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in _DATASETS[name].items():
        destination = target_dir / filename
        if destination.exists():
            print(f"'{filename}' is already present in {target_dir}.")
            continue

        print(f"Downloading '{filename}' from {url} ...")
        urlretrieve(url, destination)
        print(f"Done. '{filename}' downloaded to {destination}.")

def download_mnist_dataset(path):
    """Trigger torchvision's own download/caching for MNIST into ``path``."""
    from torchvision import datasets

    datasets.MNIST(str(path), train=True, download=True)
    datasets.MNIST(str(path), train=False, download=True)

if __name__ == "__main__":
    # By default, download all datasets next to the other notebook datasets.
    datasets_dir = Path(__file__).resolve().parent.parent
    for dataset_name in _DATASETS:
        download_dataset(dataset_name, datasets_dir)
    download_mnist_dataset(datasets_dir)