#!/usr/bin/env python3
"""Upload the Yarrowia promoter benchmarks to the HuggingFace Hub.

Requires `huggingface-cli login` first. Run:
    python src/upload_to_hf.py --user YOUR_HF_USERNAME
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True, help="your HuggingFace username")
ap.add_argument("--name", default="yarrowia-promoter-strength")
a = ap.parse_args()
repo_id = f"{a.user}/{a.name}"

create_repo(repo_id, repo_type="dataset", exist_ok=True)
HfApi().upload_folder(folder_path=str(ROOT / "hf_dataset"),
                      repo_id=repo_id, repo_type="dataset")
print(f"\nhttps://huggingface.co/datasets/{repo_id}")
