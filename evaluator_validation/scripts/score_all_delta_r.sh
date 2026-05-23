#!/bin/bash
# Score ΔR_r for every CAS / LAS / SFT / LoRA / IA3 prediction file the
# paper's Table 1 / Table 2 reports, using the trained Mistral-7B
# Bradley-Terry reward model (best so far: mistral7b_bt_pm1, test_acc=0.9646).
#
# Why "probe + skip-missing":  the user has two un-shared GPU machines that
# share an identical layout but not contents.  Each machine only holds a
# subset of the prediction files, so this script tries every expected path
# and skips the ones that don't exist on the current host.
#
# Outputs (under $OUT_ROOT):
#   per_file/<TAG>.json          per-file summary written by score_with_reward_model.py
#   delta_r_summary.json         aggregated JSON across all successfully-scored files
#   delta_r_summary.tsv          tab-separated  (paste into Excel)
#   delta_r_summary.md           markdown table (paste into the paper)
#   run.log                      full stdout/stderr trace
#
# Launch:
#   nohup bash score_all_delta_r.sh > score.log 2>&1 &
#   tail -f score.log

set -uo pipefail   # NOTE: no -e on purpose — we want to keep going past per-file failures

############################################################
# Paths (override via env if needed)
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$SCRIPT_DIR"

# Reward-model checkpoint (the SOTA mistral7b row from run_reward_models.sh)
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/evaluator_validation/models}"
RM_CKPT="${RM_CKPT:-$CKPT_ROOT/mistral7b_bt_pm1}"

# Where the project saves its training-side / eval-side prediction files.
# These mirror the layout produced by scripts/{cas,las,sft,lora,ia3}/run_*.sh.
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
CAS_RESULTS_ROOT="${CAS_RESULTS_ROOT:-$REPO_ROOT/scripts/cas/main/results}"
CAS_ABLATION_ROOT="${CAS_ABLATION_ROOT:-$REPO_ROOT/scripts/cas/ablation/results}"
LAS_SAVES_ROOT="${LAS_SAVES_ROOT:-$REPO_ROOT/scripts/saves}"
# SFT/LoRA/IA3 eval files go to NAS:
EVAL_SAVES_ROOT="${EVAL_SAVES_ROOT:-${REPO_ROOT}/eval/saves}"

# Output
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/delta_r_rm_mistral7b}"
PER_FILE_DIR="$OUT_ROOT/per_file"
mkdir -p "$PER_FILE_DIR"

# Reward-model scoring options
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LEN="${MAX_LEN:-512}"
PROMPT_MODE="${PROMPT_MODE:-}"   # leave empty -> read from $RM_CKPT/args.json

# Multi-GPU parallelism
#   Each GPU gets its own subshell that serially processes a round-robin
#   slice of CANDIDATES.  Mistral-7B in bf16 fits ~15GB on an A800, so 1
#   job per GPU is the safe default — don't oversubscribe.
#   Override NUM_GPUS / GPU_IDS to change layout.
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS_DEFAULT=""
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_IDS_DEFAULT+="${g},"
done
GPU_IDS_DEFAULT="${GPU_IDS_DEFAULT%,}"   # strip trailing comma
GPU_IDS="${GPU_IDS:-$GPU_IDS_DEFAULT}"   # e.g. "0,1"  or "0,2,3"
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARR[@]}
echo "[gpu] NUM_GPUS=$NUM_GPUS  GPU_IDS=${GPU_ID_ARR[*]}"

############################################################
# Sanity check on the reward model
############################################################
if [ ! -d "$RM_CKPT" ]; then
    echo "[fatal] reward-model dir not found: $RM_CKPT"
    echo "        Set RM_CKPT=/path/to/best_rm_dir to override."
    exit 2
fi
echo "[paths] SCRIPT_DIR        = $SCRIPT_DIR"
echo "[paths] RM_CKPT           = $RM_CKPT"
echo "[paths] CAS_RESULTS_ROOT  = $CAS_RESULTS_ROOT  (exists: $([ -d "$CAS_RESULTS_ROOT" ] && echo yes || echo no))"
echo "[paths] LAS_SAVES_ROOT    = $LAS_SAVES_ROOT    (exists: $([ -d "$LAS_SAVES_ROOT" ] && echo yes || echo no))"
echo "[paths] EVAL_SAVES_ROOT   = $EVAL_SAVES_ROOT   (exists: $([ -d "$EVAL_SAVES_ROOT" ] && echo yes || echo no))"
echo "[paths] OUT_ROOT          = $OUT_ROOT"
echo

############################################################
# Build the (TAG, METHOD, MODEL, JSONL) catalogue.
#
# TAG is the per-file output key (safe for filenames).
# Add more entries here freely; missing files are skipped automatically.
############################################################
CANDIDATES=(
    # ---- Table 1 (training-free, CAS) ----------------------------------
    "cas_qwen3b|CAS|Qwen-3B|$CAS_RESULTS_ROOT/Qwen2.5-3B-Instruct_experiment/final_predictions.jsonl"
    "cas_llama1b|CAS|Llama-1B|$CAS_RESULTS_ROOT/Llama-3.2-1B-Instruct_experiment/final_predictions.jsonl"
    "cas_mistral7b|CAS|Mistral-7B|$CAS_RESULTS_ROOT/Mistral-7B-Instruct-v0.3_experiment/final_predictions.jsonl"
    "cas_qwen7b|CAS|Qwen-7B|$CAS_RESULTS_ROOT/Qwen2.5-7B-Instruct_experiment/final_predictions.jsonl"
    "cas_qwen14b|CAS|Qwen-14B|$CAS_RESULTS_ROOT/Qwen2.5-14B-Instruct_experiment/final_predictions.jsonl"
    "cas_qwen32b|CAS|Qwen-32B|$CAS_RESULTS_ROOT/Qwen2.5-32B-Instruct_experiment/final_predictions.jsonl"

    # ---- Table 1 baselines (Few-shot / ActAdd), if you have them dumped:
    "fewshot_qwen3b|Few-shot|Qwen-3B|$CAS_ABLATION_ROOT/Qwen2.5-3B-Instruct_fewshot/final_predictions.jsonl"
    "fewshot_mistral7b|Few-shot|Mistral-7B|$CAS_ABLATION_ROOT/Mistral-7B-Instruct-v0.3_fewshot/final_predictions.jsonl"
    "fewshot_qwen7b|Few-shot|Qwen-7B|$CAS_ABLATION_ROOT/Qwen2.5-7B-Instruct_fewshot/final_predictions.jsonl"
    "actadd_qwen3b|ActAdd|Qwen-3B|$CAS_ABLATION_ROOT/Qwen2.5-3B-Instruct_actadd/final_predictions.jsonl"
    "actadd_mistral7b|ActAdd|Mistral-7B|$CAS_ABLATION_ROOT/Mistral-7B-Instruct-v0.3_actadd/final_predictions.jsonl"
    "actadd_qwen7b|ActAdd|Qwen-7B|$CAS_ABLATION_ROOT/Qwen2.5-7B-Instruct_actadd/final_predictions.jsonl"

    # ---- Table 2 (supervised, LAS / LoRA / IA3 / SFT) ------------------
    # LAS (run_affine_pipeline.sh, DATASET=all -> SUFFIX="")
    "las_llama1b|LAS|Llama-1B|$LAS_SAVES_ROOT/Llama-3.2-1B-Instruct/affine_pipeline_custom/eval_results/generated_predictions.jsonl"
    "las_qwen3b|LAS|Qwen-3B|$LAS_SAVES_ROOT/Qwen2.5-3B-Instruct/affine_pipeline_custom/eval_results/generated_predictions.jsonl"
    "las_mistral7b|LAS|Mistral-7B|$LAS_SAVES_ROOT/Mistral-7B-Instruct-v0.3/affine_pipeline_custom/eval_results/generated_predictions.jsonl"

    # LoRA (typical tag: lora_ep10_r4a8_dp -> "all" split is the un-suffixed dir,
    #       but some checkpoints use "_4all"; we probe both).
    "lora_llama1b|LoRA|Llama-1B|$EVAL_SAVES_ROOT/Llama-3.2-1B-Instruct/lora_ep10_r4a8_dp/mycode_test/generated_predictions.jsonl"
    "lora_llama1b_alt|LoRA|Llama-1B|$EVAL_SAVES_ROOT/Llama-3.2-1B-Instruct/lora_ep10_r4a8_dp_4all/mycode_test/generated_predictions.jsonl"
    "lora_qwen3b|LoRA|Qwen-3B|$EVAL_SAVES_ROOT/Qwen2.5-3B-Instruct/lora_ep10_r4a8_dp/mycode_test/generated_predictions.jsonl"
    "lora_qwen3b_alt|LoRA|Qwen-3B|$EVAL_SAVES_ROOT/Qwen2.5-3B-Instruct/lora_ep10_r4a8_dp_4all/mycode_test/generated_predictions.jsonl"
    "lora_mistral7b|LoRA|Mistral-7B|$EVAL_SAVES_ROOT/Mistral-7B-Instruct-v0.3/lora_ep10_r4a8_dp/mycode_test/generated_predictions.jsonl"
    "lora_mistral7b_alt|LoRA|Mistral-7B|$EVAL_SAVES_ROOT/Mistral-7B-Instruct-v0.3/lora_ep10_r4a8_dp_4all/mycode_test/generated_predictions.jsonl"

    # IA3
    "ia3_llama1b|IA3|Llama-1B|$EVAL_SAVES_ROOT/Llama-3.2-1B-Instruct/ia3_dp_ep10_lr1e3/eval_results/generated_predictions.jsonl"
    "ia3_llama1b_alt|IA3|Llama-1B|$EVAL_SAVES_ROOT/Llama-3.2-1B-Instruct/ia3_dp_ep10_lr1e3_4all/eval_results/generated_predictions.jsonl"
    "ia3_qwen3b|IA3|Qwen-3B|$EVAL_SAVES_ROOT/Qwen2.5-3B-Instruct/ia3_dp_ep10_lr1e3/eval_results/generated_predictions.jsonl"
    "ia3_qwen3b_alt|IA3|Qwen-3B|$EVAL_SAVES_ROOT/Qwen2.5-3B-Instruct/ia3_dp_ep10_lr1e3_4all/eval_results/generated_predictions.jsonl"
    "ia3_mistral7b|IA3|Mistral-7B|$EVAL_SAVES_ROOT/Mistral-7B-Instruct-v0.3/ia3_dp_ep10_lr1e3/eval_results/generated_predictions.jsonl"
    "ia3_mistral7b_alt|IA3|Mistral-7B|$EVAL_SAVES_ROOT/Mistral-7B-Instruct-v0.3/ia3_dp_ep10_lr1e3_4all/eval_results/generated_predictions.jsonl"

    # SFT reference (upper bound; full FT)
    "sft_llama1b|SFT_ref|Llama-1B|$EVAL_SAVES_ROOT/Llama-3.2-1B-Instruct/full_v1_ep10/mycode_test/generated_predictions.jsonl"
    "sft_llama1b_alt|SFT_ref|Llama-1B|$EVAL_SAVES_ROOT/Llama-3.2-1B-Instruct/full_v1_ep10_4all/mycode_test/generated_predictions.jsonl"
    "sft_qwen3b|SFT_ref|Qwen-3B|$EVAL_SAVES_ROOT/Qwen2.5-3B-Instruct/full_v1_ep10/mycode_test/generated_predictions.jsonl"
    "sft_qwen3b_alt|SFT_ref|Qwen-3B|$EVAL_SAVES_ROOT/Qwen2.5-3B-Instruct/full_v1_ep10_4all/mycode_test/generated_predictions.jsonl"
    "sft_mistral7b|SFT_ref|Mistral-7B|$EVAL_SAVES_ROOT/Mistral-7B-Instruct-v0.3/full_v1_ep10/mycode_test/generated_predictions.jsonl"
    "sft_mistral7b_alt|SFT_ref|Mistral-7B|$EVAL_SAVES_ROOT/Mistral-7B-Instruct-v0.3/full_v1_ep10_4all/mycode_test/generated_predictions.jsonl"

    # ---- Bonus: the locally-checked-in prediction dumps for Qwen-3B ----
    "local_cas_qwen3b|CAS|Qwen-3B(local)|$REPO_ROOT/scripts/allpreds_qwen3b/cas.jsonl"
    "local_las_qwen3b|LAS|Qwen-3B(local)|$REPO_ROOT/scripts/allpreds_qwen3b/las.jsonl"
    "local_lora_qwen3b|LoRA|Qwen-3B(local)|$REPO_ROOT/scripts/allpreds_qwen3b/lora.jsonl"
    "local_ia3_qwen3b|IA3|Qwen-3B(local)|$REPO_ROOT/scripts/allpreds_qwen3b/ia3.jsonl"
    "local_sft_qwen3b|SFT_ref|Qwen-3B(local)|$REPO_ROOT/scripts/allpreds_qwen3b/sft.jsonl"
    "local_fewshot_qwen3b|Few-shot|Qwen-3B(local)|$REPO_ROOT/scripts/allpreds_qwen3b/fewshot.jsonl"
)

############################################################
# Pre-scan: skip missing & cached entries up-front, so the
# per-GPU dispatchers only see real work.
############################################################
SKIPPED=()
CACHED=()
TODO=()

for entry in "${CANDIDATES[@]}"; do
    IFS='|' read -r TAG METHOD MODEL JSONL <<< "$entry"
    DISPLAY="${METHOD} / ${MODEL}"
    PER_FILE_OUT="$PER_FILE_DIR/${TAG}.json"

    if [ ! -f "$JSONL" ]; then
        echo "[scan] [skip]   $TAG  ($JSONL not present)"
        SKIPPED+=("$TAG|$DISPLAY|$JSONL")
        continue
    fi
    if [ -f "$PER_FILE_OUT" ] && [ -z "${FORCE:-}" ]; then
        echo "[scan] [cached] $TAG  -> $PER_FILE_OUT (FORCE=1 to redo)"
        CACHED+=("$TAG|$DISPLAY|$PER_FILE_OUT")
        continue
    fi
    echo "[scan] [todo]   $TAG  -> will score on GPU pool"
    TODO+=("$entry")
done

echo
echo "Plan: ${#TODO[@]} to score, ${#CACHED[@]} cached, ${#SKIPPED[@]} missing."
echo

############################################################
# Round-robin assign TODO across GPUs, run NUM_GPUS subshells
# in parallel.  Each subshell processes its slice serially,
# pinned to one GPU via CUDA_VISIBLE_DEVICES.
############################################################
PROMPT_MODE_ARG=()
if [ -n "$PROMPT_MODE" ]; then
    PROMPT_MODE_ARG=(--prompt_mode "$PROMPT_MODE")
fi

# Build per-GPU work files (one entry per line, "|"-separated fields).
WORK_DIR="$OUT_ROOT/_work"
mkdir -p "$WORK_DIR"
rm -f "$WORK_DIR"/work_gpu*.txt

if [ "${#TODO[@]}" -gt 0 ]; then
    for ((i=0; i<${#TODO[@]}; i++)); do
        slot=$((i % NUM_GPUS))
        echo "${TODO[$i]}" >> "$WORK_DIR/work_gpu${slot}.txt"
    done

    PIDS=()
    for ((slot=0; slot<NUM_GPUS; slot++)); do
        gpu_id="${GPU_ID_ARR[$slot]}"
        work_file="$WORK_DIR/work_gpu${slot}.txt"
        [ -f "$work_file" ] || continue
        gpu_log="$OUT_ROOT/gpu${gpu_id}.log"

        # Worker subshell — runs serially on its assigned GPU.
        (
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            # With CUDA_VISIBLE_DEVICES set, the assigned card is renumbered to
            # local index 0 inside this subshell, so always pass cuda:0.
            local_device="cuda:0"
            echo "[gpu${gpu_id}] worker started  (CUDA_VISIBLE_DEVICES=$gpu_id)"
            while IFS='|' read -r TAG METHOD MODEL JSONL; do
                [ -z "$TAG" ] && continue
                PER_FILE_OUT="$PER_FILE_DIR/${TAG}.json"
                DISPLAY="${METHOD} / ${MODEL}"
                echo "[gpu${gpu_id}] >>> $TAG  ($DISPLAY)"
                if python "$PY_DIR/score_with_reward_model.py" \
                        --rm_ckpt    "$RM_CKPT" \
                        --pred_jsonl "$JSONL" \
                        --out_json   "$PER_FILE_OUT" \
                        --label      "$DISPLAY" \
                        --batch_size "$BATCH_SIZE" \
                        --max_len    "$MAX_LEN" \
                        --device     "$local_device" \
                        "${PROMPT_MODE_ARG[@]}" ; then
                    echo "[gpu${gpu_id}] [ok]   $TAG -> $PER_FILE_OUT"
                else
                    echo "[gpu${gpu_id}] [FAIL] $TAG ($JSONL)"
                fi
            done < "$work_file"
            echo "[gpu${gpu_id}] worker done."
        ) > "$gpu_log" 2>&1 &
        pid=$!
        PIDS+=("$pid")
        echo "[launch] gpu_id=$gpu_id  pid=$pid  log=$gpu_log  ($(wc -l < "$work_file") jobs)"
    done

    echo
    echo "[wait] waiting for ${#PIDS[@]} GPU workers..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    echo "[wait] all workers done."
fi

############################################################
# Rebuild SCORED/FAILED bookkeeping from per-file JSONs.
############################################################
SCORED=("${CACHED[@]}")
FAILED=()
for entry in "${TODO[@]}"; do
    IFS='|' read -r TAG METHOD MODEL JSONL <<< "$entry"
    PER_FILE_OUT="$PER_FILE_DIR/${TAG}.json"
    DISPLAY="${METHOD} / ${MODEL}"
    if [ -f "$PER_FILE_OUT" ]; then
        SCORED+=("$TAG|$DISPLAY|$PER_FILE_OUT")
    else
        FAILED+=("$TAG|$DISPLAY|$JSONL")
    fi
done

############################################################
# Aggregate
############################################################
echo
echo "================================================================"
echo " Aggregating per-file summaries -> delta_r_summary.{json,tsv,md}"
echo "================================================================"

python - "$OUT_ROOT" "$RM_CKPT" <<'PY'
import json, os, sys, math
from pathlib import Path

out_root = Path(sys.argv[1])
rm_ckpt  = sys.argv[2]
per_dir  = out_root / "per_file"

rows = []
for p in sorted(per_dir.glob("*.json")):
    try:
        rec = json.loads(p.read_text())
    except Exception as e:
        print(f"[warn] could not parse {p}: {e}")
        continue
    if "deltaR_MAE" not in rec:
        continue
    rec["tag"] = p.stem
    rows.append(rec)

if not rows:
    print("[warn] no per-file summaries found — nothing to aggregate.")
    sys.exit(0)

# JSON
agg = {
    "rm_ckpt": rm_ckpt,
    "n_files": len(rows),
    "rows": rows,
}
(out_root / "delta_r_summary.json").write_text(json.dumps(agg, indent=2))

# TSV
def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x): return "NaN"
        return f"{x:.4f}"
    return str(x)

tsv_cols = ["tag", "label", "n",
            "mean_r_pred", "mean_r_label",
            "deltaR_MAE", "deltaR_Bias", "deltaR_RMSE",
            "pearson_r", "spearman_r"]
lines = ["\t".join(tsv_cols)]
for r in rows:
    lines.append("\t".join(_fmt(r.get(c)) for c in tsv_cols))
(out_root / "delta_r_summary.tsv").write_text("\n".join(lines) + "\n")

# Markdown
md_cols = ["tag", "label", "n",
           "ΔR (MAE)", "Bias", "Pearson r", "Spearman r"]
md = ["| " + " | ".join(md_cols) + " |",
      "|" + "|".join(["---"] * len(md_cols)) + "|"]
for r in sorted(rows, key=lambda r: (r["label"], r["tag"])):
    md.append("| " + " | ".join([
        r["tag"], r["label"], str(r["n"]),
        f"{r['deltaR_MAE']:.3f}",
        f"{r['deltaR_Bias']:+.3f}",
        f"{r['pearson_r']:.3f}",
        f"{r['spearman_r']:.3f}",
    ]) + " |")
(out_root / "delta_r_summary.md").write_text("\n".join(md) + "\n")

print("\n=== paper-ready table (markdown) ===\n")
print("\n".join(md))
print(f"\nWrote {len(rows)} rows to:")
print(f"  {out_root/'delta_r_summary.json'}")
print(f"  {out_root/'delta_r_summary.tsv'}")
print(f"  {out_root/'delta_r_summary.md'}")
PY

############################################################
# Final status report
############################################################
echo
echo "================================================================"
echo " ALL DONE"
echo "================================================================"
echo "Scored OK : ${#SCORED[@]}"
for s in "${SCORED[@]}";  do echo "   ok    $s"; done
echo "Skipped   : ${#SKIPPED[@]}  (not present on this host)"
for s in "${SKIPPED[@]}"; do echo "   skip  $s"; done
echo "Failed    : ${#FAILED[@]}"
for s in "${FAILED[@]}";  do echo "   FAIL  $s"; done
echo
echo "Per-file JSONs : $PER_FILE_DIR"
echo "Aggregated     : $OUT_ROOT/delta_r_summary.{json,tsv,md}"
