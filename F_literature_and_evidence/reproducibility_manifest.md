# Reproducibility Manifest

## Purpose

Describe the minimum artifacts needed for another researcher to understand and, where permitted, reproduce the experiments.

## 1. Project identity

| Field | Value |
|---|---|
| Project title | A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices (working title) |
| Repository URL | TBD |
| Git commit | TBD |
| Experiment date range | TBD |
| Responsible researcher | TBD |
| Paper version | TBD |

## 2. Software environment

| Component | Version/source |
|---|---|
| OS | TBD |
| Python | TBD |
| PyTorch | TBD |
| CUDA/ROCm/CPU backend | TBD |
| Detection framework | TBD |
| Mamba/SSM implementation | TBD |
| Export/runtime | TBD |
| Driver | TBD |
| Key dependencies | TBD |

## 3. Data provenance

For each dataset, record:
- Official name and URL.
- Version/release or access date.
- License/terms.
- Download method.
- Original archive checksum where practical.
- Local storage path (not necessarily committed).
- Split manifest.
- Annotation conversion script and commit.
- Class mapping.
- Filtering/exclusion rules.
- Whether any data is synthetic, real, or mixed.
- Known overlap/leakage risks.

Do not redistribute data or annotations unless the license permits it.

## 4. Model and training provenance

- Architecture/config file.
- Code commit.
- Pretrained weight source, version, and license.
- Random seeds.
- Optimizer and scheduler.
- Input resolution and augmentations.
- Batch size and epoch count.
- Training hardware and runtime.
- Checkpoint-selection rule.
- Training logs and failures.

## 5. Evaluation provenance

- Evaluation script and commit.
- Metric library/version.
- Class mapping.
- Confidence threshold and NMS settings.
- Input resolution.
- Timing protocol.
- Hardware/runtime/precision.
- Number of repetitions.
- Aggregation method.
- Per-class results.
- Final checkpoint checksum.

## 6. Artifact index

| Artifact ID | Description | Path/URL | Commit/version | Access/license | Included? |
|---|---|---|---|---|---|
| ART001 | Training config | TBD | TBD | Project code | TBD |
| ART002 | Evaluation script | TBD | TBD | Project code | TBD |
| ART003 | Model checkpoint | TBD | TBD | Checkpoint license | TBD |
| ART004 | Experiment logs | TBD | TBD | Project data | TBD |
| ART005 | Split manifest | TBD | TBD | Check dataset terms | TBD |

## 7. Reproduction instructions

Write exact commands after the implementation stabilizes. Include:
1. Environment setup.
2. Dataset acquisition and verification.
3. Annotation conversion.
4. Training.
5. Evaluation.
6. Export.
7. Edge benchmark.
8. Expected artifact names and where outputs are written.

## 8. Final reproducibility checklist

- [ ] Versions and commit hashes recorded.
- [ ] Dataset source, version, and terms recorded.
- [ ] Configurations included.
- [ ] Seeds recorded.
- [ ] Evaluation script available.
- [ ] Metric definitions documented.
- [ ] Checkpoint provenance recorded.
- [ ] Export/runtime details recorded.
- [ ] Commands tested from a clean environment.
- [ ] Data/code/weights redistribution restrictions respected.
