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
                "import os\n"
                "import subprocess\n"
                "import torch\n\n"
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
                "print(\"Using device:\", device)\n\n"
                "plot_path = \"/content/outputs/strictish_h2_d6_singlecarry_colab_metrics.png\"\n"
                "ckpt_path = \"/content/outputs/strictish_h2_d6_singlecarry_colab.pt\"\n\n"
                "cmd = [\n"
                "    \"python\",\n"
                "    \"-u\",\n"
                "    \"train_strictish_h2_d6_singlecarry_from_scratch.py\",\n"
                "    \"--device\", device,\n"
                "    \"--steps\", \"1000\",\n"
                "    \"--batch-size\", \"256\",\n"
                "    \"--lr\", \"0.02\",\n"
                "    \"--mlp-hidden\", \"1\",\n"
                "    \"--log-every\", \"1\",\n"
                "    \"--eval-samples\", \"64\",\n"
                "    \"--final-eval-samples\", \"1000\",\n"
                "    \"--target-acc\", \"1.0\",\n"
                "    \"--checkpoint\", ckpt_path,\n"
                "    \"--plot-path\", plot_path,\n"
                "]\n"
                "print(\"Running:\", \" \".join(cmd))\n"
                "env = dict(os.environ)\n"
                "env[\"PYTHONUNBUFFERED\"] = \"1\"\n\n"
                "with subprocess.Popen(\n"
                "    cmd,\n"
                "    stdout=subprocess.PIPE,\n"
                "    stderr=subprocess.STDOUT,\n"
                "    text=True,\n"
                "    bufsize=1,\n"
                "    env=env,\n"
                ") as proc:\n"
                "    assert proc.stdout is not None\n"
                "    for line in proc.stdout:\n"
                "        print(line, end=\"\", flush=True)\n"
                "    return_code = proc.wait()\n\n"
                "if return_code != 0:\n"
                "    raise RuntimeError(f\"Training failed with exit code {return_code}\")"
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
                "## Inference Playground (with logits)\n\n"
                "Enter any two 10-digit-or-smaller non-negative integers. "
                "The cell decodes autoregressively and prints top-k logits at each step."
            ),
            metadata={"id": "inference-md"},
        ),
        nbf.v4.new_code_cell(
            source=(
                "import os\n"
                "import torch\n\n"
                "from train_strictish_h2_d6_singlecarry_from_scratch import ScratchModel, Tokenizer\n\n"
                "default_ckpt = \"/content/outputs/strictish_h2_d6_singlecarry_colab.pt\"\n"
                "ckpt_path = globals().get(\"ckpt_path\", default_ckpt)\n"
                "if not os.path.exists(ckpt_path):\n"
                "    raise FileNotFoundError(\n"
                "        f\"Checkpoint not found at {ckpt_path}. Run the training cell first or set ckpt_path.\"\n"
                "    )\n\n"
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
                "tok = Tokenizer()\n"
                "model = ScratchModel(tok, init_std=0.08, mlp_hidden=1).to(device)\n"
                "state = torch.load(ckpt_path, map_location=device)\n"
                "model.load_state_dict(state[\"model_state_dict\"])\n"
                "model.eval()\n\n"
                "def tok_to_text(t):\n"
                "    if t == tok.PAD:\n"
                "        return \"PAD\"\n"
                "    if t == tok.BOS:\n"
                "        return \"BOS\"\n"
                "    if t == tok.C0:\n"
                "        return \"C0\"\n"
                "    if t == tok.EOS:\n"
                "        return \"EOS\"\n"
                "    if tok.X_BASE <= t < tok.X_BASE + 100:\n"
                "        x = t - tok.X_BASE\n"
                "        d1 = x // 10\n"
                "        d2 = x % 10\n"
                "        return f\"X[{d1},{d2}]\"\n"
                "    if tok.Y0_BASE <= t < tok.Y0_BASE + 10:\n"
                "        d = t - tok.Y0_BASE\n"
                "        return f\"Y[{d},0]\"\n"
                "    if tok.Y1_BASE <= t < tok.Y1_BASE + 10:\n"
                "        d = t - tok.Y1_BASE\n"
                "        return f\"Y[{d},1]\"\n"
                "    return f\"tok[{t}]\"\n\n"
                "def run_addition_with_logits(a, b, topk=8):\n"
                "    if not (0 <= a < 10**10 and 0 <= b < 10**10):\n"
                "        raise ValueError(\"Inputs must be in [0, 10^10).\")\n\n"
                "    seq = tok.encode_problem(a, b)\n"
                "    print(\"Initial prefix tokens:\", [tok_to_text(t) for t in seq])\n\n"
                "    step = 0\n"
                "    while len(seq) < model.max_len:\n"
                "        x = torch.tensor([seq], dtype=torch.long, device=device)\n"
                "        with torch.no_grad():\n"
                "            step_logits = model(x)[0, -1]\n\n"
                "        k = min(topk, step_logits.shape[0])\n"
                "        top_vals, top_ids = torch.topk(step_logits, k=k)\n"
                "        print(f\"\\nStep {step} (predicting token {len(seq)}):\")\n"
                "        for rank, (tid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):\n"
                "            print(f\"  {rank:>2}. {tok_to_text(tid):>10}   logit={val:8.3f}\")\n\n"
                "        next_tok = int(top_ids[0].item())\n"
                "        seq.append(next_tok)\n"
                "        step += 1\n"
                "        if next_tok == tok.EOS:\n"
                "            break\n\n"
                "    pred = tok.decode_sum(seq[model.prefix_len:])\n"
                "    exp = f\"{a + b:011d}\"\n"
                "    print(\"\\nPredicted:\", pred)\n"
                "    print(\"Expected :\", exp)\n"
                "    print(\"Correct  :\", pred == exp)\n"
                "    return pred, exp, seq\n\n"
                "a = int(input(\"Enter first number (0..9999999999): \").strip())\n"
                "b = int(input(\"Enter second number (0..9999999999): \").strip())\n"
                "_ = run_addition_with_logits(a, b, topk=8)\n"
            ),
            metadata={"id": "inference-cell"},
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
