# spot-detection-benchmark

Benchmark and useful codes of five spot detectors for smFISH-like images: **deepBlink**, **Spotiflow**,
**Piscis**, **U-FISH** and a classical **LoG** (Laplacian of Gaussian & h-maxima). 
All algorithms can be used on 2D images, stacks of 2D planes and 3D volumes.
The steps are :

1. build the training datasets (`make_datasets.py`)
2. (optional) train the deep-learning tools from scratch (`train.py`) or fine-tune an existing model (`finetuning.py`)
3. run any tool on a folder of images (`run_inference.py`)
4. score any prediction folder against ground truth (`evaluate.py`)
5. sweep each tool's threshold and keep the best one per image type (`grid_search.py`)
6. analyse the results, with ground truth (`evaluation_grid_search.ipynb`) or without
   (`compare_tools_no_groundtruth.ipynb`).

## Quick start

Every tool has its own Python environment (incompatible dependencies), managed by
[pixi](https://pixi.sh). To install all environments :

```bash
pixi install --all
```

Scripts are run from the repository root, in the tool's environment:

```bash
pixi run -e spotiflow python src/spot_detection_benchmark/run_inference.py --tool spotiflow --input_dir datasets/train_val_test_split/test --output_root results/results_inference_ufish_dataset
```

```bash
pixi run python src/spot_detection_benchmark/evaluate.py --gt_dir datasets/train_val_test_split/test --pred_dir results/results_inference_ufish_dataset/results_spotiflow --output_dir results/metrics_inference_ufish_dataset/results_spotiflow
```

`-h` on any script lists its options. On the cluster, the SLURM wrappers of `bash_scripts/` run the same
scripts.

## Repository layout

| folder | content |
|---|---|
| `src/spot_detection_benchmark/` | the code |
| `models.yaml` | which model each tool uses |
| `models/` | model weights |
| `bash_scripts/` | SLURM wrappers and pixi installation |
| `notebooks/` | analysis notebooks |
