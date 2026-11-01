First round of the frozen-model runs on the 48-task set (commit 9a83f4e,
jobs 2979823, 2979824, 2979828, 2979829, 2026-11-01). Superseded: the
sandbox of that commit returned its result as the whole stdout parsed as
JSON, so a controller that printed was floored at -300 ("Output parse
error"; 1 of 192 samples in stage1-8b, 11 of 192 in stage2-8b). Fixed in
commit bea2bf8 and re-run. Kept here: each job's summary JSON (with its
provenance block) and log; the per-sample JSONL files remain on the cluster
under ~/work/lora4vr/results/<jobname>_<jobid>/.
