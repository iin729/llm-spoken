#!/usr/bin/env python3
"""
text_eval.py

evaluationinput file (predict vs label)
 label generate O_score/level refevaluation


1. json / jsonl
2. BLEUROUGEBERTScorespoken metricsO_score/level
3. O_score/level MAE/MSE/RMSE/Pearson/SpearmanAccuracy/Macro-F1/Micro-F1/ARI/NMI
4. BLEU/ROUGE
   - HuggingFace evaluate hf_
   - LLaMA-Factory jieba + sentence_bleu + rouge_chinese llamafactory_
5. tqdm write `_eval_result` fileoverridefile
6. print summary
7. added PPL (Perplexity), Level MAE/Bias, O_score Bias
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import sacrebleu
import torch
from bert_score import score
from rouge_score import rouge_scorer, scoring
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

try:
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_chinese import Rouge

    jieba.setLogLevel(logging.CRITICAL)
except ImportError:
    jieba = None
    Rouge = None
    sentence_bleu = None
    SmoothingFunction = None

import evaluate  # noqa: E402

from .eval_oscore_level import eval_oscore_and_level
from .stat_metric8 import stat_all_metrics


def compute_bleu_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
    precisions = bleu.precisions
    return {
        "BLEU": bleu.score,
        "BLEU-1": precisions[0],
        "BLEU-2": precisions[1],
        "BLEU-3": precisions[2],
        "BLEU-4": precisions[3],
        "BLEU_BP": bleu.bp,
        "BLEU_sys_len": bleu.sys_len,
        "BLEU_ref_len": bleu.ref_len,
    }


def compute_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    aggregator.add_scores(scorer.score(reference, hypothesis))
    result = aggregator.aggregate()

    flat_scores: Dict[str, float] = {}
    for metric_name, value in result.items():
        metric_key = metric_name.upper()
        flat_scores[f"{metric_key}_precision"] = value.mid.precision
        flat_scores[f"{metric_key}_recall"] = value.mid.recall
        flat_scores[f"{metric_key}_f1"] = value.mid.fmeasure
    return flat_scores


def compute_llamafactory_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """LLaMA-Factory BLEU/ROUGEjieba + sentence_bleu + rouge_chinese"""
    if not predictions or len(predictions) != len(references):
        return {}
    if None in (jieba, Rouge, sentence_bleu, SmoothingFunction):
        warnings.warn(
            " jieba / rouge_chinese / nltk none llamafactory_* ",
            RuntimeWarning,
        )
        return {}

    rouge = Rouge()
    smoothing = SmoothingFunction().method3

    sum_bleu = 0.0
    rouge_sums = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    count = 0

    for pred, ref in zip(predictions, references):
        hyp_tokens = list(jieba.cut(pred))
        ref_tokens = list(jieba.cut(ref))

        if hyp_tokens and ref_tokens:
            rouge_scores = rouge.get_scores(" ".join(hyp_tokens), " ".join(ref_tokens))[0]
        else:
            rouge_scores = {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }

        bleu_val = sentence_bleu(
            [list(ref)],
            list(pred),
            smoothing_function=smoothing,
        )

        sum_bleu += bleu_val * 100
        for key in rouge_sums:
            rouge_sums[key] += rouge_scores[key]["f"] * 100
        count += 1

    if count == 0:
        return {}

    return {
        "llamafactory_bleu-4": sum_bleu / count,
        "llamafactory_rouge-1": rouge_sums["rouge-1"] / count,
        "llamafactory_rouge-2": rouge_sums["rouge-2"] / count,
        "llamafactory_rouge-l": rouge_sums["rouge-l"] / count,
    }


def compute_hf_corpus_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """HuggingFace evaluate BLEU/ROUGE hf_ """
    if not predictions or len(predictions) != len(references):
        return {}

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    rouge_result = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    metrics: Dict[str, float] = {}
    metrics["hf_bleu"] = bleu_result["bleu"] * 100
    precisions = bleu_result.get("precisions", [])
    for i, prec in enumerate(precisions, start=1):
        metrics[f"hf_bleu_precision{i}"] = prec * 100
    metrics["hf_bleu_brevity_penalty"] = bleu_result.get("brevity_penalty")
    metrics["hf_bleu_length_ratio"] = bleu_result.get("length_ratio")
    metrics["hf_bleu_translation_length"] = bleu_result.get("translation_length")
    metrics["hf_bleu_reference_length"] = bleu_result.get("reference_length")

    for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if key in rouge_result:
            metrics[f"hf_{key}"] = rouge_result[key] * 100

    return metrics


def compute_perplexity_metrics(predictions: List[str], model_id: str = "gpt2") -> Dict[str, float]:
    """
     PPL (Perplexity)
     Hugging Face evaluate batch and GPU
    """
    # 1. iflistreturn
    # string
    predictions = [p for p in predictions if p and len(p.strip()) > 1]
    if not predictions:
        return {"mean_perplexity": 0.0}

    # envoverride ID
    model_id = os.getenv("PPL_MODEL_ID", model_id)

    # 2. (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing PPL using model: {model_id} on {device}...")

    try:
        # module_type="metric" loadmodule
        perplexity = evaluate.load("perplexity", module_type="metric")

        # 3. PPL
        # added device=device args GPU
        results = perplexity.compute(
            predictions=predictions,
            model_id=model_id,
            add_start_token=True, #
            batch_size=16,
            device=device # <---
        )

        # evaluate return 'perplexities' (list) and 'mean_perplexity' (float)
        # mean
        return {"mean_perplexity": results["mean_perplexity"]}

    except Exception as e:
        print(f"Warning: Failed to compute perplexity: {e}")
        # return -1 None after
        return {"mean_perplexity": -1.0}


def compute_bert_scores(
        predictions: List[str],
        references: List[str],
        lang: str = "en",
        device: str | None = None,
) -> List[Tuple[float, float, float]]:
    # bertscore
    need_bert_score = os.getenv("need_bert_score", True)
    if need_bert_score in [False, "False", "false"]:
        return []

    if not predictions:
        return []
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = score(predictions, references, lang=lang, device=device, verbose=False)
    return list(zip(P.tolist(), R.tolist(), F1.tolist()))


def rankdata(values: List[float]) -> List[float]:
    n = len(values)
    sorted_pairs = sorted((val, idx) for idx, val in enumerate(values))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j
    return ranks


def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return float("nan")
    return cov / denom


def spearman_correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = rankdata(xs)
    ry = rankdata(ys)
    return pearson_correlation(rx, ry)


def compute_oscore_metrics(preds: List[float], refs: List[float]) -> Dict[str, float]:
    if not preds or len(preds) != len(refs):
        return {}
    diffs = [p - r for p, r in zip(preds, refs)]
    mae = sum(abs(d) for d in diffs) / len(diffs)
    mse = sum(d * d for d in diffs) / len(diffs)
    rmse = math.sqrt(mse)
    # Bias / Mean Error:
    bias = sum(diffs) / len(diffs)

    pearson = pearson_correlation(preds, refs)
    spearman = spearman_correlation(preds, refs)
    return {
        "O_score_MAE": mae,
        "O_score_MSE": mse,
        "O_score_RMSE": rmse,
        "O_score_Bias": bias,  # New
        "O_score_Pearson": pearson,
        "O_score_Spearman": spearman,
    }


def compute_level_metrics(preds: List[Any], refs: List[Any]) -> Dict[str, float]:
    if not preds or len(preds) != len(refs):
        return {}

    metrics: Dict[str, float] = {}
    metrics["level_accuracy"] = accuracy_score(refs, preds)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics["level_macro_F1"] = f1_score(refs, preds, average="macro", zero_division=0)
        metrics["level_micro_F1"] = f1_score(refs, preds, average="micro", zero_division=0)

    if len(preds) >= 2:
        metrics["level_ARI"] = adjusted_rand_score(refs, preds)
        metrics["level_NMI"] = normalized_mutual_info_score(refs, preds)

    # value (MAE, Bias)
    try:
        # level float rowifskip
        preds_num = [float(p) for p in preds]
        refs_num = [float(r) for r in refs]
        diffs = [p - r for p, r in zip(preds_num, refs_num)]
        metrics["level_MAE"] = sum(abs(d) for d in diffs) / len(diffs)
        metrics["level_Bias"] = sum(diffs) / len(diffs) # Bias / Mean Error:
    except (ValueError, TypeError):
        pass

    return metrics


def load_data(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        return data, "jsonl"
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON filelist[{...}]")
        return data, "json"
    raise ValueError(" .jsonl .json file")


def write_data(path: Path, data: List[Dict[str, Any]], file_type: str) -> None:
    if file_type == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


########################## predict/label
from typing import Any, Dict, Optional, Tuple, List

from typing import Any, Dict, List, Optional, Tuple


def compute_single_sample_metrics(predict: str, label: str) -> Dict[str, Any]:
    """
     predict/label class
    nonereturn None
    """
    metrics: Dict[str, Any] = {}

    # --- NLG ---
    metrics.update(compute_bleu_scores(label, predict))
    metrics.update(compute_rouge_scores(label, predict))

    # --- BERTScore ---
    bert_scores = compute_bert_scores([predict], [label])
    if bert_scores:
        p, r, f1 = bert_scores[0]
        metrics["BERTScore_P"] = p
        metrics["BERTScore_R"] = r
        metrics["BERTScore_F1"] = f1

    # --- predict ---
    spoken_pred = stat_all_metrics(text=predict) or {}
    for k, v in spoken_pred.items():
        metrics[f"spoken_pred_{k}"] = v

    o_pred = None
    level_pred = None
    if spoken_pred:
        o_pred, level_pred = eval_oscore_and_level(spoken_pred)
        metrics["O_score_pred"] = o_pred
        metrics["spoken_level_pred"] = level_pred
    else:
        metrics["O_score_pred"] = None
        metrics["spoken_level_pred"] = None

    # --- label ---
    spoken_ref = stat_all_metrics(text=label) or {}
    for k, v in spoken_ref.items():
        metrics[f"spoken_ref_{k}"] = v

    o_ref = None
    level_ref = None
    if spoken_ref:
        o_ref, level_ref = eval_oscore_and_level(spoken_ref)
        metrics["O_score_ref"] = o_ref
        metrics["spoken_level_ref"] = level_ref
    else:
        metrics["O_score_ref"] = None
        metrics["spoken_level_ref"] = None

    # --- classlistcallexception ---
    single_oscore_preds = [float(o_pred)] if o_pred is not None else []
    single_oscore_refs = [float(o_ref)] if o_ref is not None else []

    single_level_preds = [level_pred] if level_pred is not None else []
    single_level_refs = [level_ref] if level_ref is not None else []

    def safe_update(target: Dict[str, Any], fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict):
                target.update(result)
            else:
                # ifreturn dictneeds
                target[fn.__name__] = result
        except Exception as e:
            target[f"{fn.__name__}_error"] = str(e)

    if single_oscore_preds and single_oscore_refs:
        safe_update(metrics, compute_oscore_metrics, single_oscore_preds, single_oscore_refs)
    else:
        metrics["compute_oscore_metrics"] = None

    if single_level_preds and single_level_refs:
        safe_update(metrics, compute_level_metrics, single_level_preds, single_level_refs)
    else:
        metrics["compute_level_metrics"] = None

    safe_update(metrics, compute_hf_corpus_metrics, [predict], [label])
    safe_update(metrics, compute_llamafactory_metrics, [predict], [label])

    return metrics


########################## predict/label

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predict/label pairs in a JSON/JSONL file.")
    parser.add_argument("input_path", type=Path, help="evaluation json jsonl file path")
    args = parser.parse_args()

    input_path: Path = args.input_path
    data, file_type = load_data(input_path)

    valid_indices: List[int] = []
    predictions: List[str] = []
    references: List[str] = []

    for idx, entry in enumerate(data):
        pred = entry.get("predict")
        ref = entry.get("label")
        if isinstance(pred, str) and isinstance(ref, str):
            valid_indices.append(idx)
            predictions.append(pred)
            references.append(ref)
        else:
            entry.setdefault("metrics", {})
            entry["metrics"]["error"] = "Missing predict or label"

    bert_results = compute_bert_scores(predictions, references)

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    oscore_preds: List[float] = []
    oscore_refs: List[float] = []
    level_preds: List[Any] = []
    level_refs: List[Any] = []

    progress_iter = tqdm(
        enumerate(valid_indices),
        total=len(valid_indices),
        desc="Evaluating samples",
        unit="sample",
        dynamic_ncols=True,
    )

    for local_idx, data_idx in progress_iter:
        entry = data[data_idx]
        predict_text = predictions[local_idx]
        label_text = references[local_idx]

        metrics: Dict[str, Any] = {}

        metrics.update(compute_bleu_scores(label_text, predict_text))
        metrics.update(compute_rouge_scores(label_text, predict_text))

        if bert_results:
            p, r, f1 = bert_results[local_idx]
            metrics["BERTScore_P"] = p
            metrics["BERTScore_R"] = r
            metrics["BERTScore_F1"] = f1

        spoken_pred = stat_all_metrics(text=predict_text) or {}
        for k, v in spoken_pred.items():
            metrics[f"spoken_pred_{k}"] = v

        o_pred = None
        level_pred = None
        if spoken_pred:
            o_pred, level_pred = eval_oscore_and_level(spoken_pred)
            metrics["O_score_pred"] = o_pred
            metrics["spoken_level_pred"] = level_pred

        spoken_ref = stat_all_metrics(text=label_text) or {}
        for k, v in spoken_ref.items():
            metrics[f"spoken_ref_{k}"] = v

        o_ref = None
        level_ref = None
        if spoken_ref:
            o_ref, level_ref = eval_oscore_and_level(spoken_ref)
            metrics["O_score_ref"] = o_ref
            metrics["spoken_level_ref"] = level_ref

        if o_pred is not None and o_ref is not None:
            oscore_preds.append(float(o_pred))
            oscore_refs.append(float(o_ref))

        if level_pred is not None and level_ref is not None:
            level_preds.append(level_pred)
            level_refs.append(level_ref)

        entry["metrics"] = metrics

        for k, v in metrics.items():
            if is_number(v):
                metric_sums[k] += float(v)
                metric_counts[k] += 1

    averages = {
        k: (metric_sums[k] / metric_counts[k]) if metric_counts[k] else 0.0
        for k in metric_sums
    }

    summary = {
        "total_samples": len(valid_indices),
        "metrics": averages,
    }

    summary["metrics"].update(compute_oscore_metrics(oscore_preds, oscore_refs))
    summary["metrics"].update(compute_level_metrics(level_preds, level_refs))

    summary["metrics"].update(compute_hf_corpus_metrics(predictions, references))
    summary["metrics"].update(compute_llamafactory_metrics(predictions, references))
    # PPL (Perplexity)
    summary["metrics"].update(compute_perplexity_metrics(predictions))

    output_path = input_path.with_name(f"{input_path.stem}_eval_result{input_path.suffix}")
    write_data(output_path, data, file_type)

    summary_path = input_path.parent / "eval_all_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\ndoneevaluation{len(valid_indices)} ")
    print(f"write{output_path}")
    print(f"write{summary_path}")
    print("\n===== (summary) =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ===== row Excel=====
    m = summary["metrics"]
    row = [
        m.get("BLEU", 0.0), # m.get("hf_bleu", 0.0)
        m.get("BERTScore_F1", 0.0), # BERTScore_F1
        m.get("O_score_MAE", 0.0),
        m.get("O_score_Bias", 0.0),
        m.get("level_MAE", 0.0),
        m.get("level_Bias", 0.0),
        m.get("level_accuracy", 0.0),
        m.get("mean_perplexity", 0.0),
    ]
    print("\t".join(f"{v:.6f}" if isinstance(v, (int, float)) else str(v) for v in row))



if __name__ == "__main__":
    main()