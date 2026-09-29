# Flash 迭代台账

2026-09-16T02:39:44.028098+08:00

|候选|父版本|实际裁决|证据|
|---|---|---|---|
|C1|B0|REJECTED_WITH_EVIDENCE|candidates/C1/accuracy_result.json, candidates/C1/lowering_proof.json, comparisons/C1_vs_B0.json, candidates/C1/anomaly_notes.md|
|C2|B0|INAPPLICABLE_WITH_EVIDENCE|candidates/C2/accuracy.log: backend VF stack61632 >6144, vector slots237, candidates/C2/accuracy.xml: m4_n128 passed; m4_n2048 compile failed; others not executed|
|C3|B0|REJECTED_WITH_EVIDENCE|candidates/C3/accuracy_result.json, comparisons/C3_vs_B0.json, comparisons/C3_retest_vs_B0.json, candidates/C3/lowering_proof.json, candidates/C3/retest_plan.json|
|C4|B0|PROMOTED|candidates/C4/accuracy_result.json, candidates/C4/lowering_proof.json, comparisons/C4_vs_B0_repeat.json, comparisons/C4_vs_B0_initial.json, candidates/C4/decision.md|
|C5|C4|REJECTED_WITH_EVIDENCE|candidates/C5/accuracy_result.json, candidates/C5/lowering_proof.json, comparisons/C5_vs_C4.json, comparisons/C5_vs_B0.json|
|C6|C4|REJECTED_WITH_EVIDENCE|candidates/C6/accuracy_result.json, candidates/C6/lowering_proof.json, comparisons/C6_vs_C4.json, comparisons/C6_vs_B0.json|
|C7|C4|REJECTED_WITH_EVIDENCE|candidates/C7/accuracy_result.json, candidates/C7/lowering_proof.json, comparisons/C7_vs_C4.json, comparisons/C7_vs_B0.json, evidence/C7_initial_concurrency_result.json|
|C8|C4|REJECTED_WITH_EVIDENCE|candidates/C8/accuracy_result.json, candidates/C8/lowering_proof.json, comparisons/C8_vs_C4.json, comparisons/C8_vs_B0.json|
|C9|C4|REJECTED_WITH_EVIDENCE|candidates/C9/accuracy_result.json, candidates/C9/ordered_accuracy.json, candidates/C9/lowering_proof.json, comparisons/C9_vs_C4.json, comparisons/C9_vs_C8.json, profiling/C9_initial/pipe_timeline/overlap_analysis.json|
|C10|C4|REJECTED_WITH_EVIDENCE|candidates/C10/accuracy_result.json, candidates/C10/ordered_accuracy.json, candidates/C10/lowering_proof.json, comparisons/C10_vs_C8_fresh.json, comparisons/C10_vs_C4.json, comparisons/C10_vs_B0.json, candidates/C10/decision.md|

当前最佳C4；两项结构义务SATISFIED，最终search-coverage --final 已VALID。最终完整Level1与B0/B*新鲜性能复测进行中。

2026-09-16T03:08:11.008991+08:00: Final C4 pooled6 FAIL m256_n128 1.05826993, GM .87101958. C11 CREATED after source/API investigation; no data excluded.

2026-09-16T03:17:54.386697+08:00: raw frequency audit discovered 19/495 non-rated rows across completed profiling. Saved audit and prospective quality plan, no exclusions or normalization.

2026-09-16T03:31:14.434109+08:00: Before C11 timeline diagnosis, fixed own analyzer's pipeline-specific block-count assumption using actual same-source ordinary metrics. Existing C8/C9 evidence preserved; see evidence/C11_timeline_analyzer_adaptation.json.

2026-09-16T03:34:03.611486+08:00: C11 frequency retry first msprof process child failed at set_device507033/E39007 although msprof returned0; collector rejected missingOPPROF. No session2/3. Read-only healthy49C/controlCPU1 then same latest directentry9/9 succeeded. Original failure retained; new recovery plan under observed changed state, no fallback.

2026-09-16T03:50:13.914764+08:00: Post-startup recovery C4/C11 each3x9 full1650. C11gm.9198438/worst1.062884 vsC4, gainborderline8.0156% vsnoise7.0026%+1margin. Notpromoted; all6 zero-exclusion interleave plan plus freshB0 control writtenbeforecollection.

2026-09-16T04:00:57.828599+08:00: C11 RESULT rejected after all6 recovery samples, GM.973326 below noise. FreshB0 vsC4 current GM.920580/worst1.042273; enter new final attempt acceptance2, not yet VERIFIED.

2026-09-16T04:12:43.937589+08:00: acceptance2 FAIL m64_n1281.080106 despiteall54freq1650 andLevel1 90/90. C12 CREATED fromC4 with ownregister-lifetime/exp-reuse hypothesis and hardware reciprocal dependency, gateVALID beforeedit. No finalarchive/report published.

2026-09-16T04:27:04.178292+08:00: C12 39PASS/all27freq1650; GM1.014794 vsC4 andN2048 VEC~.460us vs.207, notpromotable. Before any further finalsampling fixedexact12-per-versionplan, includesallfailedacceptance2 samples, nointermediateacceptance/noexclusions; C12diagnostic/RESULT first.

- 2026-09-16T04:38:02.730311+08:00 C12补充9份trace及实际裁决完成：39精度通过，GM1.014794/最差1.261059相对C4，拒绝；当前C4继续固定12样本最终验收。
