# 最终检测报告整合

## 1. 范围说明

- 本轮只做最终检测报告整合，来源为服务器完整检测结果包。
- 解压路径：`D:\MindSpore\mindspore_badnets_gtsrb\outputs\server_detection_results_complete`
- STRIP++ protocol v3 是 sample-level detection。
- Neural Cleanse full 43-class 是 model-level detection。
- 两个检测链路都已在服务器完成。
- 旧 STRIP++ v2 screening 只作为 preliminary，不作为正式结论。
- 本结果不包含防御成功结论。

## 2. 服务器结果来源

- `outputs/server_detection_results_complete/outputs/defense/detection_protocol_v3/detection_metrics_summary.json`
- `outputs/server_detection_results_complete/outputs/defense/neural_cleanse_server/square_main/neural_cleanse_summary.json`
- `outputs/server_detection_results_complete/outputs/defense/neural_cleanse_server/checkerboard_main/neural_cleanse_summary.json`
- `outputs/server_detection_results_complete/outputs/defense/detection_audit/framework_compatibility_summary.json`
- `outputs/server_detection_results_complete/outputs/defense/detection_audit/methodology_audit_summary.json`

## 3. STRIP++ protocol v3 结果

STRIP++ protocol v3 作为正式 sample-level trigger detection 结果使用。

| Variant | detection_rate | FPR | ROC_AUC | PR_AUC | F1 | balanced_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| square_main | 0.8933 | 0.0067 | 0.9550 | 0.9643 | 0.9404 | 0.9433 |
| checkerboard_main | 1.0000 | 0.0000 | 0.9983 | 0.9993 | 1.0000 | 1.0000 |

Overall:

| mean_detection_rate | mean_FPR | mean_ROC_AUC | mean_PR_AUC |
| ---: | ---: | ---: | ---: |
| 0.9467 | 0.0033 | 0.9766 | 0.9818 |

结论：服务器复现实验中的 STRIP++ protocol v3 达到正式检测结果要求，可作为 sample-level trigger detection 的验收结果。

## 4. full 43-class Neural Cleanse 结果

full 43-class Neural Cleanse 作为正式 model-level 后门目标定位结果使用。

| Variant | completed_class_count | suspected_target_class | target_label_0_detected | mask_norm_class_0 | second_smallest_mask_norm | median_mask_norm | MAD anomaly index | anomaly_threshold | reversed_success_rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| square_main | 43 | 0 | true | 242.2166 | 499.6801 | 669.8555 | 4.5259 | 2.0 | 1.0 |
| checkerboard_main | 43 | 0 | true | 116.1734 | 379.8344 | 593.5593 | 6.3364 | 2.0 | 1.0 |

结论：两个模型的 full 43-class Neural Cleanse 均完成 43 类扫描，均成功定位 `target_label=0`，可作为 model-level 后门目标定位的正式检测结果。

## 5. Audit 结论

框架兼容性 audit：

- `framework_audit_pass = true`
- `mindspore_native_pass = true`
- `cpu_mode_pass = true`
- `ascend_ready = true`
- `pynative_supported = true`
- `graph_supported = false`
- `no_torch_dependency = true`
- `backend_abstraction_needed = true`
- `migration_risk_level = medium`

方法学 audit：

- `methodology_audit_pass = true`
- `current_formal_detection_methodology_pass = true`
- `protocol_v3_methodology_pass = true`
- `legacy_v2_methodology_pass = true`
- `protocol_v3_exists = true`
- `protocol_v3_metrics_pass = true`
- `threshold_leakage_detected = false`
- `sample_leakage_detected = false`
- `seed_cherry_picking_detected = false`
- `metric_leakage_detected = false`
- `v2_preliminary_reason = fixed-threshold holdout instability`

审计解释：旧 STRIP++ v2 screening 在方法学上保留为 preliminary 参考，但当前正式检测结论以 protocol v3 和 full 43-class Neural Cleanse 为准。

## 6. 最终结论

服务器复现实验表明，STRIP++ protocol v3 在 sample-level trigger detection 上通过验收；full 43-class Neural Cleanse 在 model-level 后门目标定位上也通过验收。两个模型均成功定位 target_label=0。检测模块可以作为项目正式结果使用。

本结论只说明检测模块有效，不代表防御模块成功。
