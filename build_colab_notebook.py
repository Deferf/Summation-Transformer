#!/usr/bin/env python3
"""Build the Colab notebook with nbformat.

This avoids hand-editing notebook JSON and keeps cell formatting stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def make_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {
            "name": "colab_train_strictish_h2_d6_singlecarry.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }

    nb.cells = [
        nbf.v4.new_markdown_cell(
            source=(
                "# Summation Transformer Colab Runner\n\n"
                "This notebook trains `train_strictish_h2_d6_singlecarry_from_scratch.py` "
                "and saves artifacts to `/content/outputs` (Colab-safe path)."
            ),
            metadata={"id": "title-cell"},
        ),
        nbf.v4.new_code_cell(
            source=(
                "import os\n\n"
                "REPO_URL = \"https://github.com/Deferf/Summation-Transformer.git\"\n"
                "REPO_DIR = \"/content/Summation-Transformer\"\n"
                "OUT_DIR = \"/content/outputs\"\n\n"
                "if not os.path.exists(REPO_DIR):\n"
                "    !git clone {REPO_URL}\n\n"
                "%cd /content/Summation-Transformer\n"
                "!pip -q install -r requirements.txt\n"
                "!mkdir -p {OUT_DIR}"
            ),
            metadata={"id": "setup-cell"},
        ),
        nbf.v4.new_code_cell(
            source=(
                "import subprocess\n"
                "import torch\n\n"
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
                "print(\"Using device:\", device)\n\n"
                "plot_path = \"/content/outputs/strictish_h2_d6_singlecarry_colab_metrics.png\"\n"
                "ckpt_path = \"/content/outputs/strictish_h2_d6_singlecarry_colab.pt\"\n\n"
                "cmd = [\n"
                "    \"python\",\n"
                "    \"train_strictish_h2_d6_singlecarry_from_scratch.py\",\n"
                "    \"--device\", device,\n"
                "    \"--steps\", \"1200\",\n"
                "    \"--batch-size\", \"256\",\n"
                "    \"--log-every\", \"100\",\n"
                "    \"--eval-samples\", \"256\",\n"
                "    \"--final-eval-samples\", \"2000\",\n"
                "    \"--target-acc\", \"0.999\",\n"
                "    \"--checkpoint\", ckpt_path,\n"
                "    \"--plot-path\", plot_path,\n"
                "]\n"
                "subprocess.run(cmd, check=True)"
            ),
            metadata={"id": "train-cell"},
        ),
        nbf.v4.new_code_cell(
            source=(
                "from IPython.display import Image, display\n\n"
                "plot_path = \"/content/outputs/strictish_h2_d6_singlecarry_colab_metrics.png\"\n"
                "display(Image(filename=plot_path))"
            ),
            metadata={"id": "show-plot-cell"},
        ),
        nbf.v4.new_markdown_cell(
            source=(
                "Optional: download artifacts from `/content/outputs/` using Colab file browser "
                "or with `files.download(...)`."
            ),
            metadata={"id": "download-note"},
        ),
    ]
    return nb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Colab notebook with nbformat")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("colab_train_strictish_h2_d6_singlecarry.ipynb"),
        help="Output notebook path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nb = make_notebook()
    nbf.validate(nb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote notebook: {args.output}")


if __name__ == "__main__":
    main()
