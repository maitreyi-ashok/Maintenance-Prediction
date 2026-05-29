
from pathlib import Path
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import sys
import os

# ---------------- Repo Details ----------------

REPO_ID = "MaitreyiAshok/maintenance-data-files"
REPO_TYPE = "dataset"

# ---------------- Dataset Path ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "engine_data.csv"

print(f"Resolved dataset path: {DATA_PATH}")

if not DATA_PATH.exists():
    print(f"ERROR: Dataset file not found -> {DATA_PATH}")
    sys.exit(1)

print(f"Dataset found: {DATA_PATH}")

# ---------------- Auth ----------------

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable not found")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# ---------------- Create Repo if Not Exists ----------------

try:

    api.repo_info(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )

    print(f"Dataset repo already exists: {REPO_ID}")

except RepositoryNotFoundError:

    print(f"Creating dataset repo: {REPO_ID}")

    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        token=HF_TOKEN
    )

except Exception as e:

    print(f"ERROR checking repository: {e}")
    sys.exit(1)

# ---------------- Upload Dataset ----------------

try:

    api.upload_file(
        path_or_fileobj=str(DATA_PATH),
        path_in_repo="engine_data.csv",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )

    print("\n🎉 Dataset uploaded successfully!")

    print(
        f"\nDataset URL:\n"
        f"https://huggingface.co/datasets/{REPO_ID}"
    )

except Exception as e:

    print(f"ERROR uploading dataset: {e}")
    sys.exit(1)

print("Dataset registration complete!")
