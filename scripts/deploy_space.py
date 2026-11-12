"""Deploy the in-browser app to a free static Hugging Face Space.

Builds web_dist/ (page + JS + ONNX model + sample video), adds the Space card
(space/README.md, which sets `sdk: static`) and uploads it all as one commit.
Static Spaces only serve files; the visitor's browser runs the model.
Needs a one-time login first:  hf auth login

Usage (from the project folder, venv active):
    python scripts/deploy_space.py                 # Space name: traffic-people-counter
    python scripts/deploy_space.py my-space-name
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_web import build  # noqa: E402


def main():
    space_name = sys.argv[1] if len(sys.argv) > 1 else "traffic-people-counter"
    dist = build()
    shutil.copy2(ROOT / "space" / "README.md", dist / "README.md")  # Space card + settings

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{space_name}"  # fails with a clear error if not logged in

    print("Uploading:")
    for f in sorted(dist.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(dist).as_posix():30} {f.stat().st_size / 1e6:6.2f} MB")

    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    # delete_patterns removes files from earlier deploys that no longer exist locally.
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=dist,
                      commit_message="Deploy in-browser app", delete_patterns=["*"])

    sub = api.space_info(repo_id).subdomain
    print(f"\nDone. Space: https://huggingface.co/spaces/{repo_id}")
    print(f"Full page:  https://{sub}.static.hf.space")


if __name__ == "__main__":
    main()
