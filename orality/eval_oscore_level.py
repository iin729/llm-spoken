# -*- coding: utf-8 -*-
"""
run12_inference.py   ——   Apply Orality Model to New Data (Spoken Only)


1. read (oral_level_model.pkl)
2. read JSON data
3. 'spoken_statics' rowfeaturesO_Scoreandetc.
4. O_score, cluster, level writeschemasave

added
- rowargs/output file
- eval_oscore_and_level(spoken_statics) function import call
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings

# sklearnwarning
warnings.filterwarnings("ignore")

# ----------------- config -----------------
# model pathfiledirdircall
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oral_level_model.pkl")

# file path
INPUT_FILE = "./../../dataset/test_train_stat.json"
# output filepath (ifdefaultfile _scored)
OUTPUT_FILE = "./../../dataset/test_train_stat_level.json"

# --------------- load ---------------
MODEL_BUNDLE = None


def load_model_bundle():
    """load"""
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH} ...")
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return MODEL_BUNDLE


# moduleafterLoad modelload
load_model_bundle()


# --------------- function ---------------
def load_data(path):
    print(f"Loading data from {path} ...")
    with open(path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_json(f, lines=True)
            data = df.to_dict(orient='records')
    return df, data # returndfdata


def extract_features(stats_dict, feat_cols_all):
    """featuresvector"""
    stats_dict = stats_dict or {}
    row_vals = []
    for col in feat_cols_all:
        val = stats_dict.get(col, 0.0)
        if val is None:
            val = 0.0
        row_vals.append(val)
    return row_vals


def _transform_feature_matrix(X_raw, model_bundle):
    """pipelinefeatures"""
    pt = model_bundle['pt']
    ss = model_bundle['ss']
    mms = model_bundle['mms']
    X_pt = pt.transform(X_raw)
    X_std = ss.transform(X_pt)
    X_scaled = mms.transform(X_std) # matrix (0-1)
    return X_scaled


def _calculate_o_scores(X_scaled, model_bundle):
    """/featuresweights O_Score"""
    feat_cols_all = model_bundle['feat_cols_all']
    feat_cols_pos = set(model_bundle['feat_cols_pos'])
    weights = model_bundle['weights']
    scores = []
    for i in range(len(X_scaled)):
        score = 0
        for j, col_name in enumerate(feat_cols_all):
            w = weights[j]
            val = X_scaled[i, j]
            if col_name in feat_cols_pos:
                score += w * val
            else:
                score += w * (1 - val)
        scores.append(score)
    return scores


def _predict_clusters(X_scaled, model_bundle):
    """+PCA+KMeans cluster"""
    weights = np.array(model_bundle['weights'])
    X_weighted = X_scaled * weights
    X_pca = model_bundle['pca'].transform(X_weighted)
    return model_bundle['kmeans'].predict(X_pca)


def _default_output_path(input_path):
    base, ext = os.path.splitext(input_path)
    ext = ext if ext else ".json"
    return f"{base}_scored{ext}"


# --------------- canfunction ---------------
def eval_oscore_and_level(spoken_statics):
    """
    spoken_statics (dict)
    (o_score, level)
    """
    model_bundle = load_model_bundle()
    feats = extract_features(spoken_statics, model_bundle['feat_cols_all'])
    X_raw = np.array([feats])
    X_scaled = _transform_feature_matrix(X_raw, model_bundle)
    o_score = float(_calculate_o_scores(X_scaled, model_bundle)[0])
    raw_cluster_id = int(_predict_clusters(X_scaled, model_bundle)[0])
    level = model_bundle['cluster_map'].get(raw_cluster_id, -1)
    return o_score, level


def run_inference(input_path=INPUT_FILE, output_path=None):
    """
    rowpipelinecan main call
    return raw_data_list
    """
    model_bundle = load_model_bundle()

    _, raw_data_list = load_data(input_path)

    # 3. features ( Spoken)
    print("Extracting features from 'spoken_statics' ...")
    feat_cols_all = model_bundle['feat_cols_all']
    process_queue = []

    for idx, item in enumerate(raw_data_list):
        if 'spoken_statics' in item and item['spoken_statics']:
            feats = extract_features(item['spoken_statics'], feat_cols_all)
            process_queue.append({
                'idx': idx,
                'feats': feats
            })

    if not process_queue:
        print("No valid 'spoken_statics' found in input file.")
        return raw_data_list

    # matrix
    X_raw = np.array([item['feats'] for item in process_queue])
    print(f"Processing {len(X_raw)} spoken samples...")

    # 4. features
    try:
        X_scaled = _transform_feature_matrix(X_raw, model_bundle)
    except Exception as e:
        print(f"Error during transform: {e}")
        print("Ensure input data has the same feature columns as training data.")
        return raw_data_list

    # 5. O_Score
    scores = _calculate_o_scores(X_scaled, model_bundle)

    # 6. class
    raw_clusters = _predict_clusters(X_scaled, model_bundle)

    # 7.
    cluster_map = model_bundle['cluster_map']
    for i, task in enumerate(process_queue):
        idx = task['idx']
        o_score = float(scores[i])
        raw_cluster_id = int(raw_clusters[i])
        level = cluster_map.get(raw_cluster_id, -1)

        if 'spoken_statics' in raw_data_list[idx]:
            raw_data_list[idx]['spoken_statics']['O_score'] = o_score
            raw_data_list[idx]['spoken_statics']['cluster'] = level # key
            raw_data_list[idx]['spoken_statics']['level'] = level # added

    # 8. savefile
    target_output = output_path or OUTPUT_FILE
    if not target_output:
        target_output = _default_output_path(input_path)

    print(f"Saving results to {target_output} ...")
    with open(target_output, 'w', encoding='utf-8') as f:
        json.dump(raw_data_list, f, ensure_ascii=False, indent=2)

    print("✅ Inference Done (Spoken Only).")
    return raw_data_list


# --------------- CLI entry ---------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply oral level model to spoken_statics data."
    )
    parser.add_argument(
        "-i", "--input",
        default=INPUT_FILE,
        help=f"input filepathdefault{INPUT_FILE}"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="output filepathdefaultconfig OUTPUT_FILEconfig _scored"
    )
    return parser.parse_args()


def main(cli_args=None):
    args = parse_args() if cli_args is None else parse_args(cli_args)
    run_inference(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
