#!/usr/bin/env python3
"""
Export cl-nagoya/ruri-pt-large to the file layout expected by memvid.

Output files:
  <cache_dir>/ruri-pt-large.onnx
  <cache_dir>/ruri-pt-large_tokenizer.json

Requirements:
  pip install "optimum[onnxruntime]" transformers huggingface_hub
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


MODEL_ID = "cl-nagoya/ruri-pt-large"
MODEL_BASENAME = "ruri-pt-large"


def default_output_dir() -> Path:
    local = Path.home() / ".cache" / "memvid" / "text-models"
    return local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cl-nagoya/ruri-pt-large to ONNX for memvid."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory to place the ONNX model and tokenizer.json",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset to force during export",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite to replace it."
        )


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        print(
            "Missing dependency. Install with:\n"
            "  pip install \"optimum[onnxruntime]\" transformers huggingface_hub",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{MODEL_BASENAME}.onnx"
    tokenizer_path = output_dir / f"{MODEL_BASENAME}_tokenizer.json"

    try:
        ensure_writable(onnx_path, args.overwrite)
        ensure_writable(tokenizer_path, args.overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="memvid-ruri-export-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Exporting {MODEL_ID} to ONNX...", file=sys.stderr)

        export_kwargs = {
            "model_name_or_path": MODEL_ID,
            "output": tmp_path,
            "task": "feature-extraction",
            "framework": "pt",
            "trust_remote_code": False,
        }
        if args.opset is not None:
            export_kwargs["opset"] = args.opset

        main_export(**export_kwargs)

        exported_model = tmp_path / "model.onnx"
        if not exported_model.exists():
            print(
                f"Export finished but {exported_model} was not created.",
                file=sys.stderr,
            )
            return 3

        shutil.copy2(exported_model, onnx_path)

        print("Downloading tokenizer.json...", file=sys.stderr)
        try:
            tokenizer_file = Path(
                hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json")
            )
            shutil.copy2(tokenizer_file, tokenizer_path)
        except EntryNotFoundError:
            print(
                "tokenizer.json is not published for this model. "
                "The ONNX model was exported, but memvid cannot load ruri-pt-large "
                "until a compatible tokenizer.json is generated.",
                file=sys.stderr,
            )

            for extra_name in ("vocab.txt", "tokenizer_config.json", "special_tokens_map.json"):
                try:
                    extra_file = Path(
                        hf_hub_download(repo_id=MODEL_ID, filename=extra_name)
                    )
                    shutil.copy2(extra_file, output_dir / extra_name)
                except EntryNotFoundError:
                    pass
            return 4

    print(f"Wrote ONNX model: {onnx_path}")
    print(f"Wrote tokenizer: {tokenizer_path}")
    print("Use TextEmbedConfig::ruri_pt_large() or TextEmbedConfig::japanese().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
