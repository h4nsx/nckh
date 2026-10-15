# Threats to Validity and Research Limitations

## Purpose

Identify risks that may make the conclusions weaker than they appear. Update this document throughout the project and discuss relevant limitations in the paper.

## 1. Dataset validity

Potential threats:
- Synthetic data may not represent real warehouse imagery.
- Dataset splits may share scenes, cameras, or near-duplicate video frames.
- Class distributions may be imbalanced.
- Annotation schemas may differ between datasets.
- Converting masks to boxes may create inconsistent labels.
- Dataset versions or hidden-test rules may change.

Mitigations:
- Document the exact dataset version and license.
- Audit annotations and class frequencies.
- Split by scene/video/camera where appropriate.
- Inspect random samples and conversion errors.
- Keep the test set isolated from model selection.
- Avoid claiming real-world generalization without real-world evaluation.

## 2. Methodological validity

Potential threats:
- A model may benefit from more parameters or training compute rather than the proposed component.
- Hyperparameter tuning may favor one method.
- Different input sizes or augmentations may make comparisons unfair.
- A small number of seeds may make differences unstable.
- Mamba implementation details may differ from the cited paper.

Mitigations:
- Use controlled ablations.
- Report model size, FLOPs/MACs, training budget, and settings.
- Preserve configuration files and commit hashes.
- State implementation deviations.
- Run multiple seeds when feasible.

## 3. Metric validity

Potential threats:
- FLOPs do not fully predict runtime.
- FPS may be measured with different timing boundaries.
- A model may have high overall mAP but poor performance on a critical class.
- Quantization may change confidence calibration or postprocessing behavior.

Mitigations:
- Report per-class metrics and relevant error examples.
- Measure latency on the target hardware.
- State preprocessing/postprocessing inclusion.
- Evaluate the exported/compressed model, not only the original checkpoint.

## 4. External validity

Potential threats:
- Results may generalize only to one synthetic dataset or one device.
- Different camera placement, lighting, motion blur, and object types may change results.
- One hardware/runtime configuration may not represent other edge devices.

Mitigations:
- Limit claims to the tested conditions.
- Describe the data domain and device.
- Add cross-scene or real-world tests only when permitted and available.
- Clearly label untested conditions as future work.

## 5. Reproducibility

Potential threats:
- Dependency versions or repositories change.
- Dataset download versions are not recorded.
- Random seeds and training configurations are missing.
- External pretrained weights have unclear provenance.

Mitigations:
- Record versions, commit hashes, configuration files, seeds, and checkpoint provenance.
- Save logs and evaluation scripts.
- Check licenses for code, weights, and data.
- Include a reproducibility manifest.

## 6. Publication and reporting bias

Potential threats:
- Reporting only successful experiments.
- Choosing metrics after seeing the results.
- Omitting negative results or failed deployment attempts.
- Overstating novelty or significance.

Mitigations:
- Preserve all runs.
- Predefine primary metrics and comparison rules.
- Report negative/neutral results when relevant.
- Phrase conclusions in proportion to the evidence.

## 7. Limitations section template

> This study has several limitations. First, [dataset limitation]. Second, [implementation or evaluation limitation]. Third, [hardware/generalization limitation]. These factors mean that the findings should be interpreted within [explicit scope]. Future work should investigate [feasible next steps].

Fill this only with limitations that actually apply to the completed study.
