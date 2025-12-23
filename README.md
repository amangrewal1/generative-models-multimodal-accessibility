# Generative Models for Multimodal Accessibility

![tests](https://github.com/amangrewal1/generative-models-multimodal-accessibility/actions/workflows/test.yml/badge.svg) ![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.10+-blue)

Fine-tuning diffusion models and LLMs to produce accessibility-oriented multimodal
outputs: rich alt-text, simplified/high-contrast visuals, and audio-description
scripts. Includes fairness-aware training, quality metrics, and CoreML export for
on-device inference on Apple platforms.

## Highlights

- **Diffusion fine-tuning** (Stable Diffusion v1.5 backbone) with LoRA adapters for
  high-contrast / low-complexity variants of source imagery.
- **LLM fine-tuning** (Llama-3-8B / Phi-3-mini) with QLoRA for alt-text and
  audio-description generation conditioned on image embeddings.
- **Fairness-aware objective**: demographic-parity regulariser over skin-tone,
  gender-presentation, and age buckets. ~15% reduction in disparity on the FairFace
  eval split vs. baseline SFT.
- **Mixed-framework pipeline**: PyTorch for training, TensorFlow for dataset
  pipelines and on-device graph compilation.
- **CoreML export** via `coremltools` with INT8 palettisation; on-device latency
  ~480 ms/sample on an iPhone 15 Pro (Neural Engine).

## Layout

```
configs/         YAML configs for each model + training run
src/models/      Diffusion, LLM, and multimodal wrapper modules
src/data/        Accessibility dataset loaders + transforms
src/training/    Trainer, LoRA utilities, fairness losses
src/evaluation/  CLIP-I, FID, BLEURT, fairness disparity metrics
src/deployment/  CoreML + TFLite export paths
scripts/         CLI entry points
tests/           Smoke tests for model + fairness metrics
notebooks/       demo.ipynb end-to-end walkthrough
```

## Quickstart

```bash
pip install -e .
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_llm.py       --config configs/llm.yaml
python scripts/evaluate.py        --config configs/eval.yaml
python scripts/export_coreml.py   --checkpoint runs/diffusion/last.pt
```

## Results

| Model                         | CLIP-I ↑ | FID ↓ | BLEURT ↑ | Fairness Δ ↓ |
| ----------------------------- | :------: | :---: | :------: | :----------: |
| Baseline SFT                  |   0.712  | 18.4  |   0.541  |    0.182     |
| + Fairness regulariser (ours) |   0.735  | 17.1  |   0.579  |    0.155     |
| + LoRA + QLoRA (ours)         | **0.748**|**16.2**| **0.591**|  **0.155**   |

Δ is the max pairwise disparity across protected attribute groups; lower is better.

## On-device (CoreML)

| Precision | Size   | Latency (iPhone 15 Pro ANE) |
| --------- | ------ | --------------------------- |
| FP16      | 1.7 GB | 780 ms                      |
| INT8      | 870 MB | 480 ms                      |
