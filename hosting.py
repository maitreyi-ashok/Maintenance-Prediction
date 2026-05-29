
from huggingface_hub import HfApi, login
import os
import sys

# ---------------- TOKEN ----------------

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found")
    sys.exit(1)

login(token=HF_TOKEN)

api = HfApi()

# ---------------- UPLOAD ----------------

SPACE_REPO_ID = "MaitreyiAshok/maintenance"

try:
    api.upload_folder(
        folder_path="maintenance_project/deployment",
        repo_id=SPACE_REPO_ID,
        repo_type="space"
    )

    print(" Deployment successful!")
    print(f" https://huggingface.co/spaces/{SPACE_REPO_ID}")

except Exception as e:
    print(f" Upload failed: {e}")
