import git
from pathlib import Path

repo = git.Repo(".", search_parent_directories=True).working_tree_dir

ARTIFACT_DIR = Path(repo, "outputs")
DATASET_DIR = Path(ARTIFACT_DIR, "datasets")
MODEL_DIR = Path(ARTIFACT_DIR, "models")
CONFIG_DIR = Path(repo, "distilrl", "configs")