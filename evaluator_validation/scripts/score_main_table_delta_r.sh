#!/bin/bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# score_main_table_delta_r.sh
#
# Score every prediction jsonl in level_results_for_main_table/ with the
# trained Mistral-7B Bradley-Terry reward model (mistral7b_bt_pm1) and
# aggregate the results by paper-table cell (Table 1 = CAS training-free,
# Table 2 = LAS supervised).
#
# Two A800s in parallel: round-robin slice of the work list, one worker
# per GPU.  Per-worker stdout goes to gpu0.log / gpu1.log so concurrent
# output doesn't mangle the main log.
#
# Expected layout on the GPU machine (where this script is run):
#
#   ${REPO_ROOT}/
#     evaluator_validation/                <-- run sh here; script + py live here
#       score_main_table_delta_r.sh        (this file)
#       score_with_reward_model.py         (single-file scorer)
#     evaluator_validation_models/
#       mistral7b_bt_pm1/                  (saved RM ckpt + args.json)
#     level_results_for_main_table/        (you uploaded this from Mac)
#       T1_*.jsonl
#       T2_*.jsonl
#       _index.md
#
# Outputs (under $OUT_ROOT, default ./delta_r_main_table/):
#   per_file/<TAG>.json                 per-file ΔR summary
#   summary_all.tsv                     one row per scored file (flat)
#   summary_grouped.tsv                 one row per (table, method, model)
#                                       with mean ± std across versions
#   paper_table1.md                     paper-ready markdown for Table 1
#   paper_table2.md                     paper-ready markdown for Table 2
#   gpu0.log / gpu1.log                 per-worker logs
#   run_status.txt                      final tally
#
# Launch:
#   cd ${REPO_ROOT}/evaluator_validation
#   nohup bash score_main_table_delta_r.sh > run.log 2>&1 &
#   tail -f run.log

set -uo pipefail

############################################################
# Paths (override via env)
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$SCRIPT_DIR"

SPOKENV2_ROOT="${SPOKENV2_ROOT:-${REPO_ROOT}}"
SRC_DIR="${SRC_DIR:-$SPOKENV2_ROOT/level_results_for_main_table}"
RM_CKPT="${RM_CKPT:-$SPOKENV2_ROOT/evaluator_validation_models/mistral7b_bt_pm1}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/delta_r_main_table}"
PER_FILE_DIR="$OUT_ROOT/per_file"

mkdir -p "$OUT_ROOT" "$PER_FILE_DIR"

############################################################
# Scoring hyperparameters
############################################################
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LEN="${MAX_LEN:-512}"
PROMPT_MODE="${PROMPT_MODE:-}"   # leave empty -> read from $RM_CKPT/args.json

############################################################
# GPU dispatcher config
############################################################
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS_DEFAULT=""
for ((g=0; g<NUM_GPUS; g++)); do GPU_IDS_DEFAULT+="${g},"; done
GPU_IDS_DEFAULT="${GPU_IDS_DEFAULT%,}"
GPU_IDS="${GPU_IDS:-$GPU_IDS_DEFAULT}"
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARR[@]}

############################################################
# Sanity checks
############################################################
[ -d "$SRC_DIR" ] || { echo "[fatal] SRC_DIR not found: $SRC_DIR"; exit 2; }
[ -d "$RM_CKPT" ] || { echo "[fatal] RM_CKPT not found: $RM_CKPT"; exit 2; }
[ -f "$PY_DIR/score_with_reward_model.py" ] || {
    echo "[fatal] score_with_reward_model.py not found next to this sh: $PY_DIR/"
    exit 2
}

echo "[paths] SRC_DIR  = $SRC_DIR"
echo "[paths] RM_CKPT  = $RM_CKPT"
echo "[paths] OUT_ROOT = $OUT_ROOT"
echo "[gpu]   NUM_GPUS = $NUM_GPUS   GPU_IDS = ${GPU_ID_ARR[*]}"
echo

############################################################
# Pre-scan: find all T1_* and T2_* prediction jsonls
############################################################
SKIPPED=()
CACHED=()
TODO=()

shopt -s nullglob
for f in "$SRC_DIR"/T1_*.jsonl "$SRC_DIR"/T2_*.jsonl; do
    [ -f "$f" ] || continue
    tag=$(basename "$f" .jsonl)
    out="$PER_FILE_DIR/${tag}.json"
    if [ -f "$out" ] && [ -z "${FORCE:-}" ]; then
        echo "[cached] $tag -> $out  (FORCE=1 to redo)"
        CACHED+=("$tag|$f|$out")
    else
        TODO+=("$tag|$f|$out")
    fi
done
shopt -u nullglob

echo
echo "Plan: ${#TODO[@]} to score, ${#CACHED[@]} cached."
echo

############################################################
# Round-robin assign + parallel dispatch
############################################################
PROMPT_MODE_ARG=()
if [ -n "$PROMPT_MODE" ]; then
    PROMPT_MODE_ARG=(--prompt_mode "$PROMPT_MODE")
fi

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

        (
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            local_device="cuda:0"   # always 0 inside the masked subshell
            echo "[gpu${gpu_id}] worker started  (CUDA_VISIBLE_DEVICES=$gpu_id)"
            while IFS='|' read -r TAG SRCF OUTF; do
                [ -z "$TAG" ] && continue
                echo "[gpu${gpu_id}] >>> $TAG"
                if python "$PY_DIR/score_with_reward_model.py" \
                        --rm_ckpt    "$RM_CKPT" \
                        --pred_jsonl "$SRCF" \
                        --out_json   "$OUTF" \
                        --label      "$TAG" \
                        --batch_size "$BATCH_SIZE" \
                        --max_len    "$MAX_LEN" \
                        --device     "$local_device" \
                        "${PROMPT_MODE_ARG[@]}" ; then
                    echo "[gpu${gpu_id}] [ok]   $TAG -> $OUTF"
                else
                    echo "[gpu${gpu_id}] [FAIL] $TAG"
                fi
            done < "$work_file"
            echo "[gpu${gpu_id}] worker done."
        ) > "$gpu_log" 2>&1 &
        pid=$!
        PIDS+=("$pid")
        n_jobs=$(wc -l < "$work_file")
        echo "[launch] gpu_id=$gpu_id  pid=$pid  log=$gpu_log  ($n_jobs jobs)"
    done

    echo
    echo "[wait] waiting for ${#PIDS[@]} GPU workers..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    echo "[wait] all workers done."
fi

############################################################
# Aggregate -> summary_all.tsv + summary_grouped.tsv + paper_table*.md
############################################################
echo
echo "================================================================"
echo " Aggregating per-cell results"
echo "================================================================"

python - "$OUT_ROOT" "$RM_CKPT" <<'PY'
import json, math, os, re, sys
from pathlib import Path
from collections import defaultdict

out_root = Path(sys.argv[1])
rm_ckpt = sys.argv[2]
per_dir = out_root / "per_file"

# -------------------------------------------------------------------------
# Filename parser: T1_CAS_Qwen-3B__v3_capped.jsonl -> (T1, CAS, Qwen-3B, v3_capped)
# -------------------------------------------------------------------------
# Tag patterns allowed:
#   T1_CAS_Qwen-3B__v1
#   T1_Few-shot_DeepSeek-R1__v1
#   T1_Few-shot_GPT-5.2__v1
#   T1_ActAdd_Mistral-7B__v1
#   T2_SFT_ref_Llama-1B__v1
#   T2_LoRA_Llama-1B__v2
#   T2_IA3_Qwen-3B__v1
#   T2_LAS_Mistral-7B__v3
# The method may contain a dash (Few-shot, SFT_ref) and the model may have a
# dash + size (Qwen-3B, Mistral-7B, DeepSeek-R1, GPT-5.2, Gemini-3-Flash).
# We match by anchoring on the version suffix `__v<rest>`.

TAG_RE = re.compile(
    r"^(?P<table>T[12])_(?P<method>.+?)_(?P<model>[A-Za-z][^_]*(?:-[^_]+)*)__(?P<ver>.+)$"
)

def parse_tag(tag: str):
    # Try strict first
    m = TAG_RE.match(tag)
    if m:
        return m.group("table"), m.group("method"), m.group("model"), m.group("ver")
    # Fallback: split on `__`
    if "__" in tag:
        head, ver = tag.rsplit("__", 1)
    else:
        head, ver = tag, "v?"
    parts = head.split("_")
    table = parts[0] if parts else "T?"
    # try to find a known model token
    model_anchors = ["Qwen-3B", "Qwen-7B", "Qwen-14B", "Qwen-32B", "Qwen-72B",
                     "Llama-1B", "Mistral-7B",
                     "DeepSeek-R1", "Gemini-3-Flash", "GPT-5.2"]
    method = "?"; model = "?"
    for a in model_anchors:
        if a in head:
            model = a
            method = head.split("_" + a)[0].split("_", 1)[1] if "_" + a in head else "?"
            break
    return table, method, model, ver


# -------------------------------------------------------------------------
# Load per-file summaries
# -------------------------------------------------------------------------
rows = []
for p in sorted(per_dir.glob("*.json")):
    try:
        rec = json.loads(p.read_text())
    except Exception as e:
        print(f"[warn] could not parse {p}: {e}")
        continue
    if "deltaR_MAE" not in rec:
        continue
    tag = p.stem
    table, method, model, ver = parse_tag(tag)
    rec["tag"] = tag
    rec["table"] = table
    rec["method"] = method
    rec["model"] = model
    rec["version"] = ver
    rows.append(rec)

if not rows:
    print("[warn] no per-file summaries — nothing to aggregate.")
    sys.exit(0)

print(f"  loaded {len(rows)} per-file results")

# -------------------------------------------------------------------------
# Flat summary: one row per file
# -------------------------------------------------------------------------
flat_cols = ["table", "method", "model", "version", "tag", "n",
             "mean_r_pred", "mean_r_label",
             "deltaR_MAE", "deltaR_Bias", "deltaR_RMSE",
             "pearson_r", "spearman_r"]

def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x): return "NaN"
        return f"{x:.4f}"
    return str(x)

lines = ["\t".join(flat_cols)]
for r in sorted(rows, key=lambda r: (r["table"], r["method"], r["model"], r["version"])):
    lines.append("\t".join(_fmt(r.get(c)) for c in flat_cols))
(out_root / "summary_all.tsv").write_text("\n".join(lines) + "\n")

# -------------------------------------------------------------------------
# Grouped summary: aggregate over versions per (table, method, model)
# -------------------------------------------------------------------------
groups = defaultdict(list)
for r in rows:
    key = (r["table"], r["method"], r["model"])
    groups[key].append(r)

def _mean_std(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)

grp_cols = ["table", "method", "model", "n_versions",
            "deltaR_MAE_mean", "deltaR_MAE_std",
            "deltaR_Bias_mean",
            "pearson_r_mean", "pearson_r_std",
            "versions"]
grp_lines = ["\t".join(grp_cols)]
group_summary = []
for key, gr in sorted(groups.items()):
    table, method, model = key
    n = len(gr)
    dmean, dstd = _mean_std([r["deltaR_MAE"] for r in gr])
    bmean, _    = _mean_std([r["deltaR_Bias"] for r in gr])
    pmean, pstd = _mean_std([r["pearson_r"] for r in gr])
    versions = ",".join(sorted(r["version"] for r in gr))
    group_summary.append({
        "table": table, "method": method, "model": model, "n": n,
        "dmean": dmean, "dstd": dstd, "bmean": bmean,
        "pmean": pmean, "pstd": pstd, "versions": versions,
        "gr": gr,
    })
    grp_lines.append("\t".join([
        table, method, model, str(n),
        _fmt(dmean), _fmt(dstd),
        _fmt(bmean),
        _fmt(pmean), _fmt(pstd),
        versions,
    ]))
(out_root / "summary_grouped.tsv").write_text("\n".join(grp_lines) + "\n")

# -------------------------------------------------------------------------
# Paper-ready markdown tables matching v9.48 Table 1 / Table 2 layout
# -------------------------------------------------------------------------
def get_row(table, method, model):
    for gs in group_summary:
        if gs["table"] == table and gs["method"] == method and gs["model"] == model:
            return gs
    return None

def cell(gs):
    if gs is None:
        return "—", "—", "—"
    # Per-version detail
    pv = ", ".join(f"{r['version']}={r['deltaR_MAE']:.3f}/{r['pearson_r']:.3f}"
                   for r in sorted(gs["gr"], key=lambda r: r["version"]))
    return f"{gs['dmean']:.3f}±{gs['dstd']:.3f}", f"{gs['pmean']:.3f}±{gs['pstd']:.3f}", pv

def paper_subscript(gs):
    """Render as 'ΔR$_{r}$' style: '0.30$_{0.81}$'"""
    if gs is None: return "—"
    return f"{gs['dmean']:.2f}$_{{{gs['pmean']:.2f}}}$"

# --- Table 1 ---
md1 = []
md1.append("# Paper Table 1 — CAS (training-free) ΔR$_r$ scoring")
md1.append("")
md1.append(f"Scored with: `{rm_ckpt}`")
md1.append("")
md1.append("Each cell shows `ΔR_mean±std (n versions)` and `pearson r mean±std`,")
md1.append("plus a per-version breakdown.  The compact form `ΔR$_{r}$` matches")
md1.append("how the paper renders the column.")
md1.append("")
md1.append("## Closed-source Few-shot ceilings")
md1.append("")
md1.append("| Model | ΔR$_r$ (paper-style) | ΔR mean±std | r mean±std | per-version |")
md1.append("|---|---|---|---|---|")
for model in ["DeepSeek-R1", "Gemini-3-Flash", "GPT-5.2"]:
    gs = get_row("T1", "Few-shot", model)
    d, p, pv = cell(gs)
    md1.append(f"| {model} | {paper_subscript(gs)} | {d} | {p} | {pv} |")

md1.append("")
md1.append("## Open-source backbones × {Few-shot, ActAdd, CAS}")
md1.append("")
md1.append("| Backbone | Method | ΔR$_r$ (paper-style) | ΔR mean±std | r mean±std | per-version |")
md1.append("|---|---|---|---|---|---|")
for model in ["Qwen-3B", "Mistral-7B", "Qwen-7B"]:
    for method in ["Few-shot", "ActAdd", "CAS"]:
        gs = get_row("T1", method, model)
        d, p, pv = cell(gs)
        md1.append(f"| {model} | {method} | {paper_subscript(gs)} | {d} | {p} | {pv} |")
(out_root / "paper_table1.md").write_text("\n".join(md1) + "\n")

# --- Table 2 ---
md2 = []
md2.append("# Paper Table 2 — LAS (supervised) ΔR$_r$ scoring")
md2.append("")
md2.append(f"Scored with: `{rm_ckpt}`")
md2.append("")
md2.append("| Backbone | Method | ΔR$_r$ (paper-style) | ΔR mean±std | r mean±std | per-version |")
md2.append("|---|---|---|---|---|---|")
for model in ["Llama-1B", "Qwen-3B", "Mistral-7B"]:
    for method in ["SFT_ref", "LoRA", "IA3", "LAS"]:
        gs = get_row("T2", method, model)
        d, p, pv = cell(gs)
        md2.append(f"| {model} | {method} | {paper_subscript(gs)} | {d} | {p} | {pv} |")
(out_root / "paper_table2.md").write_text("\n".join(md2) + "\n")

print()
print("=" * 64)
print("Aggregation written:")
print(f"  flat        : {out_root/'summary_all.tsv'}        ({len(rows)} rows)")
print(f"  grouped     : {out_root/'summary_grouped.tsv'}    ({len(group_summary)} cells)")
print(f"  paper T1    : {out_root/'paper_table1.md'}")
print(f"  paper T2    : {out_root/'paper_table2.md'}")
print("=" * 64)
print()
print("--- paper_table1.md ---")
print((out_root / "paper_table1.md").read_text())
print("--- paper_table2.md ---")
print((out_root / "paper_table2.md").read_text())
PY

############################################################
# Final status
############################################################
{
    echo "==== run_status.txt @ $(date) ===="
    echo "RM_CKPT  : $RM_CKPT"
    echo "SRC_DIR  : $SRC_DIR"
    echo "OUT_ROOT : $OUT_ROOT"
    echo "scored (incl cached): $(ls "$PER_FILE_DIR" 2>/dev/null | wc -l)"
    echo "todo this run       : ${#TODO[@]}"
    echo "cached at start     : ${#CACHED[@]}"
    echo "GPU IDs             : ${GPU_ID_ARR[*]}"
    echo
    echo "==== per-file outputs ===="
    ls -lh "$PER_FILE_DIR" 2>/dev/null
} > "$OUT_ROOT/run_status.txt"

echo
echo "================================================================"
echo " ALL DONE"
echo "================================================================"
echo "  per-file   : $PER_FILE_DIR/"
echo "  flat tsv   : $OUT_ROOT/summary_all.tsv"
echo "  grouped tsv: $OUT_ROOT/summary_grouped.tsv"
echo "  Table 1 md : $OUT_ROOT/paper_table1.md"
echo "  Table 2 md : $OUT_ROOT/paper_table2.md"
echo "  GPU logs   : $OUT_ROOT/gpu*.log"
echo "  status     : $OUT_ROOT/run_status.txt"
