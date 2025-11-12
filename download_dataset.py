import kagglehub
import shutil
from config import *

def download_dataset() -> str:
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")

    print("Path to dataset files:", path)
    return path

def move_dataset_from_cache_to_folder(path_to_cache: str, path_to_folder: str) -> None:
    shutil.copytree(path_to_cache, path_to_folder, dirs_exist_ok=True)
    shutil.rmtree(path_to_folder / "data")

if __name__ == "__main__":
    path_to_cache = download_dataset()
    move_dataset_from_cache_to_folder(path_to_cache=path_to_cache, path_to_folder=DATA_RAW_DIR)

