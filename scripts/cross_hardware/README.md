# Cross-Hardware Generality Experiment (Ascend 910B)

Frozen-rule cross-hardware prediction for the paper's four models
(BERT-like, GPT-2-like, ResNet-50, ViT-B/16) at batch 4.

## Files

- `predict_paper_models.py` — local PeakAware simulation-only prediction
  (Ascend910B chip config, all-save plan peak). Run on the A6000 machine:
  ```
  python scripts/cross_hardware/predict_paper_models.py --model bert_base --batch 4
  python scripts/cross_hardware/predict_paper_models.py --model gpt2 --batch 4
  python scripts/cross_hardware/predict_paper_models.py --model resnet50 --batch 4
  python scripts/cross_hardware/predict_paper_models.py --model vit_b_16 --batch 4
  ```
  Requires `PEAKAWARE_COSTMODEL_HARDWARE` (set automatically to `Ascend910B`).

- `ascend_paper_models.py` — eager full-step peak measurement on Ascend 910B
  (`torch.npu.max_memory_allocated`). Run on the Ascend server:
  ```
  python ascend_paper_models.py --model bert_base --batch 4   # etc.
  ```
  Requires torch 2.9.0+cpu + torch_npu 2.9.0rc1 + transformers 4.57.6 +
  torchvision 0.24.0+cpu (matching torch 2.9.0).

- `a6000_paper_models.py` — identical measurement on NVIDIA A6000
  (`torch.cuda.max_memory_allocated`), same model definitions.

- `symmetric_measure.py` — symmetric cross-platform measurement of one step
  (`--device cuda|npu`) used to diagnose the allocator-visible peak difference.

- `results_20260819.json` — consolidated 3-way table
  (A6000 measured / Ascend measured / simulation prediction + composition).

## Headline result

| Workload | A6000 MB | Ascend MB | Predicted MB | APE vs Ascend |
|---|---|---|---|---|
| BERT-like-2L-64H (S32) | 58.89 | 66.40 | 66.91 | 0.76% |
| GPT-2-like-2L-64H (S32) | 225.39 | 208.38 | 209.12 | 0.35% |
| ResNet-50 (224²) | 520.82 | 508.18 | 530.10* | 4.3%* |
| ViT-B/16 (224²) | 1741.37 | 1732.53 | 1843.14 | 6.38% |

\* storage-lifetime core after removing the labeled conv-backward workspace
heuristic (117.4 MB); the raw prediction with that add-on is 647.54 MB and
over-counts the A6000 as well (24.3%), i.e. the gap is a config-sensitive
workspace heuristic, not a cross-hardware modeling failure.

## Hardware configs

The Ascend910B chip/topology configs live in
`Costmodel/atencost/backend/analytical_model/hardware/`
(`chip_configs/Ascend910B.json`, `topo_configs/Ascend910B.json`).
`peakaware/cost/legacy_adapter.py` auto-selects them when
`torch.npu.is_available()` is true.
