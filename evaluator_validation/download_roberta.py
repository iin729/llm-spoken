"""Download roberta-base from HuggingFace Hub to a local directory.

Usage:
    python scripts/evaluator_validation/download_roberta.py
    # or with a custom destination:
    python scripts/evaluator_validation/download_roberta.py \
        --dest roberta-base

After it finishes you'll have:
    roberta-base/
        config.json
        tokenizer.json
        merges.txt
        vocab.json
        pytorch_model.bin   (or model.safetensors)
        ...
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="roberta-base",
                    help="HuggingFace repo id.")
    ap.add_argument("--dest",
                    default="roberta-base",
                    help="Local directory to save the model in.")
    ap.add_argument("--revision", default=None,
                    help="Optional specific git revision/branch/tag.")
    ap.add_argument("--allow_patterns", nargs="*", default=None,
                    help='File patterns to download, e.g. "*.json" "*.safetensors". '
                         'Default = everything.')
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: `huggingface_hub` is not installed. Install with:\n"
              "    pip install -U huggingface_hub", file=sys.stderr)
        sys.exit(1)

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[download] repo     = {args.repo}")
    print(f"[download] revision = {args.revision or 'main'}")
    print(f"[download] dest     = {dest}")
    print()

    local_dir = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(dest),
        local_dir_use_symlinks=False,        # copy, not symlink (NAS-safe)
        allow_patterns=args.allow_patterns,  # None = all files
        max_workers=4,
    )
    print(f"\n[done] Files saved under: {local_dir}")
    print("       You can now point ROBERTA_PATH at this directory in the sh runner.")


if __name__ == "__main__":
    main()
