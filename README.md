# Unsupervised Image Segmentation for Procedural Level Generation in Games

MSc Computer Science research, University of the Witwatersrand.

This project builds a procedural content generation pipeline that learns game assets and how they fit together **directly from level images**. It uses unsupervised object detection to find assets of any size, then extends Wave Function Collapse (WFC) to generate new levels. No predefined tile grid or hand-written adjacency rules are needed.

It builds on my [Honours research](https://github.com/Shakeel-21/Honours-Research), which used an autoencoder to extract fixed-size tiles. This work removes the fixed-tile limit.

<!-- TODO: add a pipeline diagram here -->
<!-- ![pipeline](docs/pipeline.png) -->

## Why this matters

Standard WFC needs a tileset and adjacency rules written by hand for each game. That is slow and does not transfer between games. This pipeline learns both from screenshots, so one method can work across different visual styles.

## How it works

1. **Asset extraction.** CutLER (unsupervised object detection, run through Detectron2) finds core and complex assets of arbitrary size in level images.
2. **Adjacency inference.** Two-Stage Masked Adjacency Extraction works out which assets can sit next to each other, using masks so that background does not distort the rules.
3. **Background rules.** A Ghost Pass infers background rules. <!-- TODO: one sentence on what it does and why it is needed -->
4. **Generation.** WFC extended with Non-Uniform Tiles generates new levels from the learned assets and rules. Atomic Blueprint Stamping places complex multi-tile assets as single units. <!-- TODO: one sentence on why stamping helps -->

## Results

<!-- TODO: add your real numbers. Examples of what to include:
- Games tested
- Metrics used (e.g. SSIM, tile or asset accuracy, generation success rate)
- Comparison against the Honours baseline
-->

| Game | Metric | Result |
|------|--------|--------|
| TODO | TODO | TODO |

Example outputs:

<!-- ![segmentation](docs/segmentation.png) -->
<!-- ![generated](docs/generated.png) -->

## Tech stack

Python, PyTorch, CUDA, Detectron2, CutLER, OpenCV, scikit-learn, Pillow, imagehash

## Getting started

### Requirements

- Python 3.x <!-- TODO: exact version -->
- A CUDA-capable GPU (needed for CutLER / Detectron2)

### Install

```bash
git clone https://github.com/Shakeel-21/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

<!-- TODO: add Detectron2 / CutLER install steps and where to download weights -->

### Run

```bash
# TODO: replace with your real commands
python extract_assets.py --input data/levels --output out/assets
python generate.py --assets out/assets --output out/generated
```

## Project structure

```
TODO: paste a short tree of your main folders and files
```

## Status

Work in progress. Thesis due 2027. <!-- TODO: update as you go -->

## Related work

- [Honours research: autoencoder-based tile extraction + WFC](https://github.com/Shakeel-21/Honours-Research)
- CutLER: Wang et al., *Cut and Learn for Unsupervised Object Detection and Instance Segmentation*
- Wave Function Collapse: Maxim Gumin

## Author

Shakeel Malagas, [LinkedIn](https://www.linkedin.com/in/shakeel-malagas-21907233b)
