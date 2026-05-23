#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

# =============================================================================
# score_cas_ablation_table.sh
# =============================================================================
# For every CAS-ablation variant in ./results/abl_<variant>_<MODEL>/, run two
# scoring passes and aggregate into a paper-ready Table 3 markdown.
#
# Pass A (sequential) -- eval_predictions.py
#   Computes BERTScore, level accuracy, level bias, etc.  Writes:
#       ./results/abl_<variant>_<MODEL>/eval_all_metrics.json
#
# Pass B (2-GPU parallel) -- score_with_reward_model.py
#   Computes ΔR (MAE between predicted-reward and gt-reward) and pearson r,
#   using the trained Mistral-7B Bradley-Terry RM.  Writes:
#       ./results/abl_<variant>_<MODEL>/delta_r.json
#
# Aggregation
#   ./ablation_table_results/ablation_summary.tsv
#   ./ablation_table_results/paper_table3.md      (rows ordered to match v9.48
#                                                  Table 3: CAS, Iso, -Calib,
#                                                  -Multi-layer, Random layers,
#                                                  Random vectors).
#
# Launch:
#   cd ${REPO_ROOT}/cas/ablation
#   nohup bash score_cas_ablation_table.sh > score_ablation.log 2>&1 &
#   tail -f score_ablation.log
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

############################################################
# Paths (override via env if your GPU layout differs)
############################################################
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
OUT_DIR="${OUT_DIR:-$HERE/ablation_table_results}"
mkdir -p "$OUT_DIR"

# eval_predictions.py lives in the repo's cas/eval (sibling-up of cas_paper_ablation/)
EVAL_PY_CANDIDATES=(
    "${EVAL_PY:-}"
    "$HERE/../cas/eval/eval_predictions.py"
    "$HERE/../eval/eval_predictions.py"
    "$HERE/../../scripts/cas/eval/eval_predictions.py"
    "${REPO_ROOT}/orality/eval_predictions.py"
    "${REPO_ROOT}/orality/eval_predictions.py"
    "${REPO_ROOT}/orality/eval_predictions.py"
)
EVAL_PY=""
for p in "${EVAL_PY_CANDIDATES[@]}"; do
    [ -z "$p" ] && continue
    if [ -f "$p" ]; then EVAL_PY="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"; break; fi
done

# score_with_reward_model.py lives in evaluator_validation/
RM_SCORER_CANDIDATES=(
    "${RM_SCORER:-}"
    "$HERE/../evaluator_validation/score_with_reward_model.py"
    "${REPO_ROOT}/evaluator_validation/score_with_reward_model.py"
)
RM_SCORER=""
for p in "${RM_SCORER_CANDIDATES[@]}"; do
    [ -z "$p" ] && continue
    if [ -f "$p" ]; then RM_SCORER="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"; break; fi
done

# RM checkpoint
RM_CKPT="${RM_CKPT:-${REPO_ROOT}/evaluator_validation/models/mistral7b_bt_pm1}"

# Hyperparameters for RM scorer
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LEN="${MAX_LEN:-512}"
PROMPT_MODE="${PROMPT_MODE:-}"        # empty -> read from $RM_CKPT/args.json

# GPU dispatcher
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS_DEFAULT=""
for ((g=0; g<NUM_GPUS; g++)); do GPU_IDS_DEFAULT+="${g},"; done
GPU_IDS_DEFAULT="${GPU_IDS_DEFAULT%,}"
GPU_IDS="${GPU_IDS:-$GPU_IDS_DEFAULT}"
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARR[@]}

############################################################
# Sanity checks
############################################################
fatal() { echo "[fatal] $*"; exit 2; }
[ -d "$RESULTS_DIR" ] || fatal "RESULTS_DIR not found: $RESULTS_DIR"
[ -n "$EVAL_PY"     ] && [ -f "$EVAL_PY" ]   || fatal "eval_predictions.py not found (set EVAL_PY=...)"
[ -n "$RM_SCORER"   ] && [ -f "$RM_SCORER" ] || fatal "score_with_reward_model.py not found (set RM_SCORER=...)"
[ -d "$RM_CKPT" ] || fatal "RM ckpt not found: $RM_CKPT (set RM_CKPT=...)"

EVAL_DIR="$(dirname "$EVAL_PY")"
RM_SCORER_DIR="$(dirname "$RM_SCORER")"

echo "[paths] HERE          = $HERE"
echo "[paths] RESULTS_DIR   = $RESULTS_DIR"
echo "[paths] OUT_DIR       = $OUT_DIR"
echo "[paths] EVAL_PY       = $EVAL_PY"
echo "[paths] RM_SCORER     = $RM_SCORER"
echo "[paths] RM_CKPT       = $RM_CKPT"
echo "[gpu]   NUM_GPUS=$NUM_GPUS  GPU_IDS=${GPU_ID_ARR[*]}"
echo

############################################################
# Discover variants
############################################################
echo "Discovering ablation variants under $RESULTS_DIR ..."
VARIANTS=()
shopt -s nullglob
for d in "$RESULTS_DIR"/abl_*; do
    [ -d "$d" ] || continue
    pred="$d/final_predictions.jsonl"
    if [ -f "$pred" ]; then
        tag="$(basename "$d")"
        tag="${tag#abl_}"
        rows=$(wc -l < "$pred" | awk '{print $1}')
        VARIANTS+=("$tag|$d|$pred")
        echo "  [ok]   $tag  (rows=$rows)"
    else
        echo "  [skip] $(basename "$d")  (no final_predictions.jsonl)"
    fi
done
shopt -u nullglob
echo "Total variants: ${#VARIANTS[@]}"
echo
[ "${#VARIANTS[@]}" -gt 0 ] || fatal "no ablation variants found"

############################################################
# Phase 1: eval_predictions.py  (sequential; BERTScore is fast enough)
############################################################
echo "================================================================"
echo " Phase 1: BERT / L-Acc / level metrics (eval_predictions.py)"
echo "================================================================"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r TAG VDIR PRED <<< "$entry"
    summary="$VDIR/eval_all_metrics.json"
    if [ -f "$summary" ] && [ -z "${FORCE:-}" ]; then
        echo "  [cached] $TAG  -> $summary  (FORCE=1 to redo)"
        continue
    fi
    echo ">>> $TAG"
    # eval_predictions.py uses cwd-relative imports for eval_oscore_level / stat_metric8.
    (
        cd "$EVAL_DIR"
        PYTHONPATH="$EVAL_DIR" python "$EVAL_PY" "$PRED"
    ) || echo "  [FAIL] eval $TAG"
done
echo

############################################################
# Phase 2: ΔR_r scoring  (2-GPU parallel)
############################################################
echo "================================================================"
echo " Phase 2: ΔR_r scoring with Mistral-7B BT RM ($NUM_GPUS GPUs)"
echo "================================================================"

WORK_DIR="$OUT_DIR/_work"
mkdir -p "$WORK_DIR"
rm -f "$WORK_DIR"/work_gpu*.txt

idx=0
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r TAG VDIR PRED <<< "$entry"
    rm_out="$VDIR/delta_r.json"
    if [ -f "$rm_out" ] && [ -z "${FORCE:-}" ]; then
        echo "  [cached] $TAG  -> $rm_out  (FORCE=1 to redo)"
        continue
    fi
    slot=$((idx % NUM_GPUS))
    echo "$TAG|$PRED|$rm_out" >> "$WORK_DIR/work_gpu${slot}.txt"
    idx=$((idx+1))
done

PROMPT_MODE_ARG=()
[ -n "$PROMPT_MODE" ] && PROMPT_MODE_ARG=(--prompt_mode "$PROMPT_MODE")

PIDS=()
for ((slot=0; slot<NUM_GPUS; slot++)); do
    work_file="$WORK_DIR/work_gpu${slot}.txt"
    [ -f "$work_file" ] || continue
    gpu_id="${GPU_ID_ARR[$slot]}"
    gpu_log="$OUT_DIR/gpu${gpu_id}.log"
    n_jobs=$(wc -l < "$work_file" | awk '{print $1}')

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        local_device="cuda:0"
        echo "[gpu${gpu_id}] worker started  ($n_jobs jobs)"
        while IFS='|' read -r TAG PRED OUTF; do
            [ -z "$TAG" ] && continue
            echo "[gpu${gpu_id}] >>> $TAG"
            if python "$RM_SCORER" \
                    --rm_ckpt    "$RM_CKPT" \
                    --pred_jsonl "$PRED" \
                    --out_json   "$OUTF" \
                    --label      "$TAG" \
                    --batch_size "$BATCH_SIZE" \
                    --max_len    "$MAX_LEN" \
                    --device     "$local_device" \
                    "${PROMPT_MODE_ARG[@]}" ; then
                echo "[gpu${gpu_id}] [ok]   $TAG"
            else
                echo "[gpu${gpu_id}] [FAIL] $TAG"
            fi
        done < "$work_file"
        echo "[gpu${gpu_id}] worker done"
    ) > "$gpu_log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    echo "[launch] gpu_id=$gpu_id  pid=$pid  log=$gpu_log  ($n_jobs jobs)"
done

if [ "${#PIDS[@]}" -gt 0 ]; then
    echo
    echo "[wait] waiting for ${#PIDS[@]} GPU workers..."
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[wait] all done."
fi
echo

############################################################
# Phase 3: aggregate -> ablation_summary.tsv + paper_table3.md
############################################################
echo "================================================================"
echo " Phase 3: aggregate -> ablation_summary.tsv + paper_table3.md"
echo "================================================================"

python - "$RESULTS_DIR" "$OUT_DIR" <<'PY'
import json, math, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

# Collect per-variant rows
rows = []
for vdir in sorted(results_dir.glob("abl_*")):
    if not vdir.is_dir():
        continue
    pred = vdir / "final_predictions.jsonl"
    if not pred.exists():
        continue
    tag = vdir.name
    if tag.startswith("abl_"):
        tag = tag[len("abl_"):]
    row = {"variant": tag, "predictions": str(pred)}

    em = vdir / "eval_all_metrics.json"
    if em.exists():
        try:
            m = json.loads(em.read_text()).get("metrics", {})
            row["BERTScore_F1"]   = m.get("BERTScore_F1")
            row["level_accuracy"] = m.get("level_accuracy")
            row["level_Bias"]     = m.get("level_Bias")
            row["level_MAE"]      = m.get("level_MAE")
            row["O_score_MAE"]    = m.get("O_score_MAE")
        except Exception as e:
            print(f"  [warn] could not parse {em}: {e}")

    dr = vdir / "delta_r.json"
    if dr.exists():
        try:
            d = json.loads(dr.read_text())
            row["deltaR_MAE"]  = d.get("deltaR_MAE")
            row["deltaR_Bias"] = d.get("deltaR_Bias")
            row["pearson_r"]   = d.get("pearson_r")
            row["spearman_r"]  = d.get("spearman_r")
            row["n_scored"]    = d.get("n")
        except Exception as e:
            print(f"  [warn] could not parse {dr}: {e}")

    rows.append(row)

# Ordering matches v9.48 Table 3 reading order
ORDER = ["full_cas", "iso_cas", "no_calib", "no_multilayer",
         "random_layers", "random_vec", "cas_capped", "actadd"]
def order_key(r):
    t = r["variant"]
    for i, canon in enumerate(ORDER):
        if t.startswith(canon):
            return (i, t)
    return (999, t)
rows.sort(key=order_key)

# --- Flat TSV ---
cols = ["variant", "BERTScore_F1", "level_accuracy", "level_Bias", "level_MAE",
        "O_score_MAE", "deltaR_MAE", "deltaR_Bias", "pearson_r", "spearman_r",
        "n_scored"]
def fmt(x):
    if x is None: return "—"
    if isinstance(x, float):
        if math.isnan(x): return "NaN"
        return f"{x:.4f}"
    return str(x)
lines = ["\t".join(cols)]
for r in rows:
    lines.append("\t".join(fmt(r.get(c)) for c in cols))
(out_dir / "ablation_summary.tsv").write_text("\n".join(lines) + "\n")

# --- Display labels matching paper Table 3 ---
DISPLAY = {
    "full_cas":      "CAS (full)",
    "iso_cas":       "Iso  (isotonic α(t))",
    "no_calib":      "− Calibration",
    "no_multilayer": "− Multi-layer",
    "actadd":        "ActAdd (Turner et al.)",
    "random_layers": "Random layers",
    "random_vec":    "Random vectors",
    "cas_capped":    "CAS capped (α-safe envelope)",
}
def disp(tag):
    for canon, label in DISPLAY.items():
        if tag.startswith(canon):
            return label
    return tag

# --- Paper-ready Table 3 markdown (v9.48 layout) ---
md = []
md.append("# Paper Table 3 — CAS-component ablation (auto-generated)")
md.append("")
md.append("Subscripts mirror v9.48 paper conventions:")
md.append("  • L-Acc$_{bias}$  = level accuracy with signed level bias subscript")
md.append("  • ΔR$_{r}$       = MAE between predicted-reward and gt-reward, with")
md.append("                    pearson r between r(predict) and r(label) as subscript")
md.append("")
md.append("Reward model: `mistral7b_bt_pm1` (test pair-acc 0.9646, see eval validation appendix).")
md.append("")
md.append("| Variant | BERT$\\uparrow$ | L-Acc$_{bias}$ $\\uparrow$ | ΔR$_{r}$ $\\downarrow$ |")
md.append("|---|---|---|---|")
for r in rows:
    name = disp(r["variant"])

    bert = "—"
    if isinstance(r.get("BERTScore_F1"), float):
        bert = f"{r['BERTScore_F1']:.3f}"

    la = "—"
    if isinstance(r.get("level_accuracy"), float):
        la = f"{r['level_accuracy']:.3f}"
        b = r.get("level_Bias")
        if isinstance(b, float):
            la = f"{la}$_{{{b:+.2f}}}$"

    dr = "—"
    if isinstance(r.get("deltaR_MAE"), float):
        dr = f"{r['deltaR_MAE']:.2f}"
        p = r.get("pearson_r")
        if isinstance(p, float):
            dr = f"{dr}$_{{{p:.2f}}}$"

    md.append(f"| {name} | {bert} | {la} | {dr} |")

md.append("")
md.append("**Per-variant raw values (sanity):**")
md.append("")
md.append("| Variant | BERT | L-Acc | L-Bias | ΔR | r |")
md.append("|---|---|---|---|---|---|")
for r in rows:
    md.append("| " + " | ".join([
        disp(r["variant"]),
        fmt(r.get("BERTScore_F1")),
        fmt(r.get("level_accuracy")),
        fmt(r.get("level_Bias")),
        fmt(r.get("deltaR_MAE")),
        fmt(r.get("pearson_r")),
    ]) + " |")
(out_dir / "paper_table3.md").write_text("\n".join(md) + "\n")

print()
print("=" * 64)
print("Aggregation written:")
print(f"  flat tsv  : {out_dir / 'ablation_summary.tsv'}  ({len(rows)} variants)")
print(f"  paper md  : {out_dir / 'paper_table3.md'}")
print("=" * 64)
print()
print("--- paper_table3.md ---")
print((out_dir / "paper_table3.md").read_text())
PY

echo
echo "================================================================"
echo " ALL DONE"
echo "================================================================"
echo "  TSV   : $OUT_DIR/ablation_summary.tsv"
echo "  MD    : $OUT_DIR/paper_table3.md"
echo "  GPU logs : $OUT_DIR/gpu*.log"
echo
echo "Per-variant artifacts (alongside each abl_<variant>/ dir):"
echo "  eval_all_metrics.json    (BERT, L-Acc, level metrics)"
echo "  delta_r.json             (ΔR, pearson r, n_scored)"
