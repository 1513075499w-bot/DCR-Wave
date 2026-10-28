import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/dcr_wave_slim_structure.json").read_text())
    backbone = cfg["backbone"]
    print("DCR-Wave slim_w0667: configuration-derived architecture summary")
    print("Input: B x 3 x 640 x 640")
    print("Stage  Blocks  Channels  Height  Width")
    for stage, (depth, dim) in enumerate(zip(backbone["depths"], backbone["dims"]), 1):
        size = backbone["img_size"] // (4 * 2 ** (stage - 1))
        print(f"{stage:5d}  {depth:6d}  {dim:8d}  {size:6d}  {size:5d}")
    print("Stage shapes are configuration-derived, before each stage's downsampling.")
    print("Decoder input: B x 512 x 20 x 20; decoder not bundled.")
    print("Model, train.py, and evaluate.py are included; external runtime, experiment YAML, and decoder implementations are required.")


if __name__ == "__main__":
    main()
