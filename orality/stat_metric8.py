import json
import re
import spacy
import tqdm
import math
from .ideadensity import depid

# Load spaCy model (disable NER, keep tagger and parser)

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ==========================================
# 1. User-defined oral lexicon
# ==========================================
from .vocabulary import FILLER_WORDS, INTERACTION_MARKERS, STANCE_AND_HEDGES, LEXICAL_SIMPLIFICATION, ALL_VOCAB

# ==========================================
# 2. Core computation functions
# ==========================================
def count_phrase_frequency(text_lower, phrase_set):
    """
    Count total occurrences (instance count) and unique types of a phrase set in text.
    Use regex \b boundary to avoid matching inside words (e.g. 'ash' inside 'wash').
    """
    total_count = 0
    unique_types = 0

    # Length-desc ordering matters for regex replacementbut for pure looped matching,
    # as long as the source string is untouchedorder only affects WHO is counted firstnot whether it's counted.
    # overlapping counts are allowed(e.g. both 'you know' and 'you know what' fire)acceptable for feature engineering.

    for phrase in phrase_set:
        # escape regex chars
        escaped_phrase = re.escape(phrase)
        # word boundary
        pattern = r'\b' + escaped_phrase + r'\b'

        # count all instances of this phrase
        matches = re.findall(pattern, text_lower)
        count = len(matches)

        if count > 0:
            total_count += count
            unique_types += 1  # record hit of a new unique type

    return total_count, unique_types


def calculate_refined_lexical_density(doc):
    """
    Compute Noun-Verb Ratio (NVR) and effective word count
    """
    total_valid_tokens = 0
    noun_count = 0
    verb_count = 0

    # 1. iterate tokens
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        total_valid_tokens += 1
        pos = token.pos_

        # count base POS
        if pos in ["NOUN", "PROPN"]:
            noun_count += 1
        elif pos in ["VERB", "AUX"]:
            verb_count += 1

    # --- aggregate ---
    nv_ratio = noun_count / (verb_count + 1)
    return nv_ratio, total_valid_tokens


def calculate_pid(text):
    # 1. prepare vocab
    all_vocab = " ".join(ALL_VOCAB).lower()
    all_vocab_set = set(all_vocab.split(" "))  # set for O(1) lookup

    for i in ["i", "me", "my", "mine","you","your", "i'm",
              "you're", "he's", "she's", "it's", "we're", "they're",
              "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd","i'll", "you'll"]:
        if i in all_vocab_set:
            all_vocab_set.remove(i)

    # 2. filter logic: drop tokens in oral/filler vocab (return False)
    def custom_token_filter(token):
        # if token is in oral/filler vocab, return False (drop, do not count toward Idea)
        if token.lemma_.lower() in all_vocab_set:
            # print(f"{token.lemma_.lower()} in all_vocab_set")
            return False
            # otherwise keep
        return True

    # 3. depid call arg tuning


    density, word_count, dependencies = depid(text,custom_token_filters=[custom_token_filter])
    return density


def calculate_repeated_ratio(text: str):
    """
    Compute strict repetition ratio.
    define
    1. (: "go go")
    2. >=2 (: "abc def abc def")
    3. >=3 (: "one two three ... one two three")

    return (, , )
    """
    # 1.
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    total = len(tokens)

    if total < 2:
        return 0.0

    # set“”set
    repeated_indices = set()

    # --- A: ( 1 and 2) ---
    # 1 () and 2 () can
    # length L then tokens[i:i+L] == tokens[i+L:i+2*L]
    # L 1 total // 2

    max_len = total // 2
    for length in range(1, max_len + 1):
        # iteratecan
        # needs i + 2*length
        for i in range(total - 2 * length + 1):
            chunk1 = tuple(tokens[i: i + length])
            chunk2 = tuple(tokens[i + length: i + 2 * length])

            if chunk1 == chunk2:
                # ifset
                for k in range(i, i + 2 * length):
                    repeated_indices.add(k)

    # --- B: ( 3) ---
    # 3: >=3
    # iflength4length3
    # “override”needslength 3 3-gramscan
    # 3-gram >1 override

    if total >= 3:
        trigram_positions = {} # : { ('word1', 'word2', 'word3'): [start_index1, start_index2, ...] }

        # 3-gram
        for i in range(total - 2):
            trigram = tuple(tokens[i: i + 3])
            if trigram not in trigram_positions:
                trigram_positions[trigram] = []
            trigram_positions[trigram].append(i)

        # 1 3-gram
        for trigram, starts in trigram_positions.items():
            if len(starts) > 1:
                # 3-gram
                for start_idx in starts:
                    repeated_indices.add(start_idx)
                    repeated_indices.add(start_idx + 1)
                    repeated_indices.add(start_idx + 2)

    # --- ---
    repeated_count = len(repeated_indices)
    ratio = repeated_count / total if total > 0 else 0.0

    # # 2. features (Feature Transformation)
    # #
    # # 1e-9 0 butaftercansqrt(0)=0butepsilon
    # import numpy as np
    # ratio = np.sqrt(ratio)

    return ratio


def calculate_score_with_root_normalization(count, unique_types, total_words):
    """
    helperafter
    (Count / Sqrt(N)) * (1 + 0.1 * Unique_Types)
    """
    if total_words == 0:
        return 0.0

    # 1. ()
    # sqrt
    density = count / math.sqrt(total_words)

    # 2. (optionalclass)
    # class 10% weights 3 fillers 1.3
    diversity_bonus = 1 + (0.1 * unique_types)

    return density
    # return density * diversity_bonus



def stat_all_metrics(text):
    """
    functionreturnfeatures
    """
    if not text or not isinstance(text, str):
        #
        return {k: 0.0 for k in
                ["Fillers", "Interaction", "Stance & Hedges", "Simplification", "Repetition", "PID", "NVR"]}

    text_lower = text.lower()

    # 2. Spacy (done)
    # notenlp()canASRcan
    doc = nlp(text)
    # 3. Spacy
    nv_ratio, total_words = calculate_refined_lexical_density(doc)


    #
    if total_words == 0:
        return {k: 0.0 for k in
                ["Fillers", "Interaction", "Stance & Hedges", "Simplification", "Repetition", "PID", "NVR"]}

    # --- modify count and unique_types ---
    filler_count, filler_unique = count_phrase_frequency(text_lower, FILLER_WORDS)
    interact_count, interact_unique = count_phrase_frequency(text_lower, INTERACTION_MARKERS)
    stance_count, stance_unique = count_phrase_frequency(text_lower, STANCE_AND_HEDGES)
    simplify_count, simplify_unique = count_phrase_frequency(text_lower, LEXICAL_SIMPLIFICATION)

    repeat_freq = calculate_repeated_ratio(text)
    pid_density = calculate_pid(text)

    # --- modify ---
    # notevalue 0-1 Scorevalue 0.0 - 5.0
    return {
        # ()
        "Fillers": round(calculate_score_with_root_normalization(filler_count, filler_unique, total_words), 4),
        "Interaction": round(calculate_score_with_root_normalization(interact_count, interact_unique, total_words), 4),
        "Stance & Hedges": round(calculate_score_with_root_normalization(stance_count, stance_unique, total_words), 4),
        "Simplification": round(calculate_score_with_root_normalization(simplify_count, simplify_unique, total_words),
                                4),

        # needsunchangedcanor
        # unchanged
        "Repetition": round(repeat_freq, 4),

        # ()
        "PID": round(pid_density, 4),
        "NVR": round(nv_ratio, 4),
    }

# ==========================================
# 3. filetest
# ==========================================

def process_file(inpath, outpath):
    print(f"Start processing: {inpath}")
    num_text = 0
    with open(inpath, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        while first_char and first_char.isspace():
            first_char = f.read(1)
        f.seek(0)
        is_list_file = first_char == '['

        if is_list_file:
            data_list = json.load(f)
            for i, data in tqdm.tqdm(enumerate(data_list)):
                spoken = data["spoken"]
                written = data["written"]

                data["spoken_statics"] = stat_all_metrics(spoken)
                data["written_statics"] = stat_all_metrics(written)

                num_text += 1
                if i % 500 == 0:
                    print(f"has processed {i}")

            with open(outpath, "w", encoding="utf-8") as fw:
                json.dump(data_list, fw, ensure_ascii=False)
            return data_list
        else:
            with open(outpath, "w", encoding="utf-8") as fw:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    spoken = data["spoken"]
                    written = data["written"]

                    data["spoken_statics"] = stat_all_metrics(spoken)
                    data["written_statics"] = stat_all_metrics(written)

                    fw.write(json.dumps(data, ensure_ascii=False) + "\n")
                    num_text += 1
                    if i % 500 == 0:
                        print(f"has processed {i}")

    print(f"Done! Saved to {outpath}")


if __name__ == "__main__":
    # --- test ---
    test_cases = [
        # Case 1: (High Orality)
        "Um, I mean, it's kinda like, you know, really hard to get things done when you're tired and stuff.",

        # Case 2: (Real Spoken)
        "I still think that for us that's too low I mean location wise it's perfect We've updated a lot of things.",

        # Case 3: (Written)
        "The proposed price is too low considering the ideal location and the extensive updates.",

        # Case 4: (High Literacy)
        "The subsequent implementation of the protocol facilitated a significant reduction in latency."
    ]

    print(f"{'Metric':<15} | {'Oral(Fake)':<10} | {'Oral(Real)':<10} | {'Written':<10} | {'Formal':<10}")
    print("-" * 65)

    results = [stat_all_metrics(t) for t in test_cases]

    #
    POS_FEATS = ['Fillers', 'Interaction', 'Stance & Hedges', 'Simplification', 'Repetition']
    NEG_FEATS = ['PID', 'NVR']

    keys = POS_FEATS + NEG_FEATS

    for k in keys:
        row = f"{k:<15} | "
        for res in results:
            val = res.get(k, 0)
            row += f"{val:<10} | "
        print(row)


    # iffileneedsnote
    # process_file(".jsonl", "output_with_metrics.jsonl")
    inpath = "./../../dataset/test_sft.json"
    outpath = "./../../dataset/test_sft_stat.json"
    data_list = process_file(inpath, outpath)
    for i in data_list:
        i["type"] = "test"

    inpath = "./../../dataset/train_sft.json"
    outpath = "./../../dataset/train_sft_stat.json"
    data_list_train = process_file(inpath, outpath)
    for i in data_list_train:
        i["type"] = "train"

    data_all = data_list + data_list_train

    all_outpath = "./../../dataset/test_train_stat.json"
    with open(all_outpath, "w", encoding="utf-8") as fw:
        json.dump(data_all, fw, ensure_ascii=False)