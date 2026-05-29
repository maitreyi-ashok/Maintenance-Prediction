
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import sys
import os

from pathlib import Path

# ---------------- Repo Details ----------------

REPO_ID = "MaitreyiAshok/maintenance-data-files"

REPO_TYPE = "dataset"

DATA_PATH = "maintenance_project/data/engine_data.csv"


# ---------------- Auth ----------------
# SECURITY: replace hardcoded token with secret — see Known Issues section

api = HfApi(token=os.getenv("HF_TOKEN"))

if not os.path.exists(DATA_PATH):

    print(f"ERROR: Dataset file not found -> {DATA_PATH}")

    sys.exit(1)

print(f"Dataset found: {DATA_PATH}")
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
        private=False
    )

except Exception as e:

    print(f"ERROR checking repository: {e}")

    sys.exit(1)

# ---------------- Upload Dataset ----------------
try:

    api.upload_file(

        path_or_fileobj=DATA_PATH,

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
