# boundary-weighted-active-learning-
Active learning for annotation-efficient abdominal CT segmentation with MedSAM, using a boundary-weighted acquisition criterion that multiplies ensemble variance by mean-prediction gradient magnitude.

markdown
# Boundary-Weighted Active Learning for Abdominal CT Segmentation

This repository contains the training and evaluation pipeline for the paper
**"Boundary-Aware Bayesian Active Learning for Annotation-Efficient Abdominal
CT Segmentation with Foundation Models."**

The method fine-tunes a LoRA-adapted MedSAM prompt decoder on abdominal CT
slices and runs an active learning loop in which the acquisition function
combines ensemble predictive variance with the gradient magnitude of the mean
prediction. This *boundary-weighted* criterion preferentially queries slices
whose epistemic uncertainty is concentrated at sharp anatomical edges, which is
precisely where new annotations most reduce downstream surface error.

## Overview

Foundation models such as MedSAM provide strong promptable segmentation out of
the box, yet their generic features still require domain-specific fine-tuning
for small, low-contrast abdominal structures such as the pancreas, adrenal
glands, and peripancreatic vessels. The dominant cost of fine-tuning is expert
annotation. Classical acquisition functions — entropy, variance, and gradient
magnitude used in isolation — fail to beat random selection on this task,
because they ignore the geometric structure of the segmentation loss. The
boundary-weighted criterion resolves this by multiplying the two signals at the
pixel level.

## Repository Contents

    run_medsam_al.py       Training, evaluation, active learning, and figures
    README.md              This file
    LICENSE                MIT License

## Requirements

The pipeline requires Python 3.10 or later with PyTorch. A CUDA-capable GPU is
strongly recommended; a CPU-only run is possible but takes 40 to 60 times
longer.

Install the dependencies:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.38.0 nibabel scipy pandas matplotlib Pillow psutil

If you do not have a CUDA-capable GPU, replace the first line with:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

and pass `--device cpu` to the script.

## Data

The pipeline reads pairs of single-slice NIfTI files from two directories:

    <data_root>/image_001.nii.gz, image_002.nii.gz, ...
    <label_root>/label_001.nii.gz, label_002.nii.gz, ...

Each label file is a multi-class mask with class indices:

    0  background
    1  liver
    2  spleen
    3  right kidney
    4  pancreas
    5  adrenal gland
    6  peripancreatic vessel

The pipeline was developed on the public Abdominal CT Scans dataset:

> Gut, Daniel (2021), "Abdominal CT scans", Mendeley Data, V1,
> doi: 10.17632/6x684vg2bg.1

The dataset is released under Creative Commons Attribution 4.0 International
(CC BY 4.0). The 575 axial slices used in the paper are the same files
distributed with that release.

## Usage

### Development run

A fast smoke test that trains for a single seed and two labelling budgets. It
should complete in 20 to 45 minutes on a mid-range GPU and produces the same
CSV columns as the full run:

    python run_medsam_al.py --dev \
        --data_root /path/to/images \
        --label_root /path/to/labels \
        --out_dir /path/to/outputs

### Full run

The configuration reported in the paper uses five random seeds, five labelling
budgets, and an ensemble of five decoders. The full run takes approximately 60
to 100 GPU-hours:

    python run_medsam_al.py \
        --data_root /path/to/images \
        --label_root /path/to/labels \
        --out_dir /path/to/outputs

### Command-line arguments

    --data_root    Directory containing image_*.nii(.gz). Default: E:\images
    --label_root   Directory containing label_*.nii(.gz). Default: E:\labels
    --out_dir      Output directory for CSV, checkpoints, and figures.
                   Default: E:\figures
    --dev          Reduced run: 1 seed, 1 epoch, K=2, 2 budgets
    --device       cuda or cpu. Default: cuda

## Method

The pipeline trains an ensemble of K LoRA-adapted MedSAM prompt decoders while
keeping the vision encoder frozen. For each query round it scores every
unlabeled slice with one of five acquisition functions:

| Acquisition | Criterion |
|-------------|-----------|
| Random | uniform random |
| Entropy | mean predictive entropy |
| Variance-only | total ensemble variance |
| Gradient-only | mean-prediction gradient magnitude |
| Boundary-weighted | pixelwise product of variance and gradient magnitude |

The top-scoring slices are added to the labelled set, the ensemble is retrained,
and Dice, HD95, and ASSD are evaluated on a fixed held-out test set.

### Metrics

- Dice similarity coefficient, computed per class and averaged across the three
  hard organs (pancreas, adrenal gland, peripancreatic vessel).
- 95th-percentile Hausdorff distance, normalised by the image diagonal.
- Average symmetric surface distance, normalised by the image diagonal.

### Ablations

The script supports the following ablations, each of which requires a separate
run with a modified configuration:

- Ensemble size K. Set `ENSEMBLE_K` in the script to 1, 3, 5, or 7 and rerun.
- Weighting parameters α and β. Modify the `acq_score` function to accept
  non-unit exponents and rerun.
- Bounding-box perturbation. Modify `BOX_NOISE_PX` and rerun.

## Outputs

After a full run, the output directory contains:

    results.csv             One row per (seed, method, budget)
    figures/fig01_*.png      Representative axial slices with GT overlays
    figures/fig02_*.png      Active learning curves
    figures/fig03_*.png      Final round bar chart
    ...
    figures/fig13_*.png      Per-class Dice heatmap

The CSV is the single source of truth for every number that appears in the
paper. Regenerating the figures from the same CSV always produces the same
results.

## Reproducibility

The pipeline is deterministic given the same inputs and seed. Setting the
environment variable `PYTHONHASHSEED=0` before running removes the last source
of run-to-run variation.

    $env:PYTHONHASHSEED=0
    python run_medsam_al.py --dev

The MedSAM weights are loaded from the Hugging Face Hub identifier
`flaviagiammarino/medsam-vit-base`. The exact weights used in the paper are
recorded in the checkpoint files.
