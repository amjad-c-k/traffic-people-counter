"""Deploy the Gradio app to a Hugging Face Space.

Collects only what the Space needs (app code, the OpenVINO INT8 model, the sample
video and the Space config files in space/) into a temporary folder and uploads
it as one commit. Needs a one-time login first:  hf auth login

Usage (from the project folder, venv active):
    python scripts/deploy_space.py                 # Space name: traffic-people-counter
    python scripts/deploy_space.py my-space-name
"""
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "yolo26n_int8_openvino_model"

# (source in this repo, destination inside the Space)
FILES = [
    ("app.py", "app.py"),
    ("counter/__init__.py", "counter/__init__.py"),
    ("counter/detector.py", "counter/detector.py"),
    ("counter/lap_fallback.py", "counter/lap_fallback.py"),
    ("counter/line_counter.py", "counter/line_counter.py"),
    ("counter/draw.py", "counter/draw.py"),
    ("counter/pipeline.py", "counter/pipeline.py"),
    ("samples/highway.mp4", "samples/highway.mp4"),
    ("space/README.md", "README.md"),            # Space card + settings (SDK, Python version...)
    ("space/requirements.txt", "requirements.txt"),
    ("space/packages.txt", "packages.txt"),      # system libraries installed with apt
]


def main():
    space_name = sys.argv[1] if len(sys.argv) > 1 else "traffic-people-counter"
    if not MODEL_DIR.exists():
        sys.exit(f"Missing {MODEL_DIR.relative_to(ROOT)} - run scripts/export_models.py first.")

    api = HfApi()
    user = api.whoami()["name"]  # fails with a clear error if you are not logged in
    repo_id = f"{user}/{space_name}"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for src, dst in FILES:
            (stage / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, stage / dst)
        shutil.copytree(MODEL_DIR, stage / "models" / MODEL_DIR.name)

        print("Uploading:")
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(stage).as_posix():55} {f.stat().st_size / 1e6:6.2f} MB")

        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=stage,
                          commit_message="Deploy app with OpenVINO INT8 model")

    print(f"\nDone. Space: https://huggingface.co/spaces/{repo_id}")
    print("The first build takes a few minutes (installing packages).")


if __name__ == "__main__":
    main()
