# Manuscript-to-code alignment

| Manuscript component | Implementation |
| --- | --- |
| CGSFNet stem, four stages, AdaLSConv, and Stage 4 MSA | `src/pgrcl/model.py` |
| Normalized 512-dimensional identity representation | `CGSFNet.forward_features` |
| Stage-specific prototype and historical identity reference, Eqs. (6)–(7) | `PrototypeMemory` and `construct_stage_prototypes` |
| Classification, feature-distillation, historical-prototype, and relation-preservation losses, Eqs. (8)–(12) | `src/pgrcl/losses.py` |
| LwF logit distillation, Eqs. (13)–(14) | `logit_distillation_loss` and `train_stage` |
| Accuracy, macro Precision/Recall/F1, ACPV, and Forgetting | `src/pgrcl/metrics.py` |
| T1 initialization and T2–T4 continual updates | `Experiment.run_continual` |
| EWC, LwF, Replay, Fine-tuning, and Joint Training | `src/pgrcl/experiment.py` and `src/pgrcl/engine.py` |
| Contextual-kernel and Stage 4 MSA ablations | `scripts/run_backbone_ablations.py` |
| PGR-CL loss-weight sensitivity | `scripts/run_sensitivity.py` |
| Cross-version prototype compatibility | `scripts/prototype_compatibility.py` |
| Seed-matched paired analysis | `scripts/paired_statistics.py` |
| Acquisition-controlled test subset | `scripts/evaluate_subset.py` |
| Historical-data storage calculation | `scripts/storage_report.py` |

The code does not include the separately published LBL-SheepNet and GAN-CGSFR implementations. Those methods were retrained from their original codebases for the direct comparison reported in the manuscript; replacing them with approximate reimplementations here would not preserve the stated evaluation protocol.
