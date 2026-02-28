from __future__ import annotations

import argparse

import torch

from addition_data import AdditionTokenizer
from model import DecoderOnlyTransformer, ModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 10-digit addition inference from a trained checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--a", type=int, required=True, help="First addend in [0, 10^10)")
    p.add_argument("--b", type=int, required=True, help="Second addend in [0, 10^10)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not (0 <= args.a < 10**10 and 0 <= args.b < 10**10):
        raise ValueError("Both numbers must be in [0, 10^10)")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    stoi = checkpoint["tokenizer_stoi"]
    tokenizer = AdditionTokenizer(stoi=stoi, itos={i: ch for ch, i in stoi.items()})
    model_cfg = ModelConfig(**checkpoint["model_config"])

    device = torch.device(args.device)
    model = DecoderOnlyTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = f"^{args.a:010d}+{args.b:010d}="
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=12)
    text = tokenizer.decode(output[0].tolist())

    eq_idx = text.find("=")
    if eq_idx == -1:
        predicted = ""
    else:
        dollar_idx = text.find("$", eq_idx + 1)
        predicted = text[eq_idx + 1 : dollar_idx if dollar_idx != -1 else len(text)]
    expected = f"{(args.a + args.b):011d}"

    print(f"prompt:     {prompt}")
    print(f"generated:  {text}")
    print(f"predicted:  {predicted}")
    print(f"expected:   {expected}")
    print(f"correct:    {predicted == expected}")


if __name__ == "__main__":
    main()
