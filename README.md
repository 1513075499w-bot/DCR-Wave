# DCR-Wave

DCR-Wave is a research implementation for event-based object detection.

This repository provides the model architecture and selected core components. 

## Files

- `dcr_wave/backbone.py`: DCR-Wave backbone and stage assembly.
- `dcr_wave/wave_core.py`: wave propagation, contour graph building, graph encoding, and temporal features.
- `dcr_wave/detector.py`: a generic detector wrapper (the training script uses the external runtime's DETR wrapper).
- `configs/dcr_wave_slim_structure.json`: model structure and decoder interface configuration.
- `train.py`: training entry point using this repository's backbone and a compatible external detection runtime.
- `evaluate.py`: COCO bbox evaluation entry point using the external runtime and a compatible checkpoint.
- `tools/inspect_structure.py`: prints the configuration-derived stage shapes.
- `requirements-structure.txt`: dependencies for the model structure; **not** a complete training environment.

## Installation

Install the required dependencies:

`python -m pip install -r requirements-structure.txt`

## Datasets

Our experiments use the following event-based object detection datasets:

- **EvDET200K**: A benchmark covering 10 object categories.
  Dataset information and download links are available from the
  [official project page](https://github.com/Event-AHU/OpenEvDET).

- **GEN1**: An automotive detection dataset with annotations for cars and pedestrians.
  Please visit the
  [official dataset page](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)
  for details and download instructions.

## Usage

To display the model architecture:

`python tools/inspect_structure.py`

## Train and evaluate

Replace the paths below with actual files from your environment. The recipe
must define the dataset paths, dataloaders, transforms, losses, optimizer,
postprocessor, and other runtime settings. These examples do not run from this
repository alone.

First, run a real-data preflight test without updating model parameters:

```bash
python train.py --runtime /path/to/compatible-runtime --recipe /path/to/experiment.yml --check --check-batch-size 1
```

Train:

```bash
python train.py --runtime /path/to/compatible-runtime --recipe /path/to/experiment.yml --output output/train
```

Evaluate a checkpoint:

```bash
python evaluate.py --runtime /path/to/compatible-runtime --recipe /path/to/experiment.yml --checkpoint output/train/best.pth --output output/evaluation
```

## Release Scope

This repository provides a partial research-code release for architecture inspection and further development.