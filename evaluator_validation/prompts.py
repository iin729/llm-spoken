"""Prompt template for LLM-as-judge pairwise spoken-vs-written comparison.

The first block (`GUIDELINES`) mirrors Listing 2 in the paper verbatim and MUST
NOT be edited. The second block adds (a) per-level calibration examples and
(b) three pairwise few-shot demos using `=>` to mark the expected answer. The
final query asks the model to emit JSON only after a final `=>`.

All few-shot examples are real, short utterances drawn from the WSPairs training
set; their levels are the g(·) (EWM+KMeans) labels.
"""

# ----------------------------------------------------------------------------
# (1) Original annotation guidelines — VERBATIM, DO NOT EDIT.
# ----------------------------------------------------------------------------
GUIDELINES = r"""Spoken-vs-Written Compare Annotation Guidelines

Task
Given two English variants of the same content, choose which one is more spoken. ``More spoken" means more spontaneous, conversational delivery while preserving all facts and intended meaning.

Core signals of ``more spoken"
- Structure and density: Shorter clauses, sentence splitting, looser syntax, lower lexical density. Written style favors complex subordination and dense noun phrases.
- Contractions and simpler wording: I'm we're it's; simpler verbs and phrasing over formal alternatives.
- Fillers and discourse markers: um, uh, hmm; well, so, okay, look, anyway, you know. These organize turns and soften entries.
- Stance and hedging: I think, I guess, maybe, kind of, sort of, probably. Shows speaker stance/uncertainty.
- Interaction cues: you know?, right?, yeah?, direct address (you), brief acknowledgments (yeah, mm-hm). Signals audience engagement or turn-yielding.
- Repetition and repairs: small natural repeats, restarts, self corrections (The price is it's about), echoing words for emphasis. Should feel purposeful, not random noise.
- Fragmentation and rhythm: Sentence fragments, trailing off, incremental delivery (Okay, so the plan). Still coherent at the discourse level.
- Prosody-like punctuation: Ellipses and dashes can mark spoken rhythm when used sparingly.

Reference scale (for intuition, not labeling)
- Level 0 (written): Formal structure, high lexical density, no fillers, no repetition.
- Level 1 (spoken, slight): Light contractions, mild simplification, rare markers.
- Level 2 (spoken, low): Occasional fillers (um/uh/okay), well/so, slight sentence splitting.
- Level 3 (spoken, medium): Clear conversational flow, some hedges (I think/maybe), more fragmentation, small natural repetition.
- Level 4 (spoken, high): Frequent interaction cues (you know/right?), more hedging, noticeable repetition, shorter clauses, still coherent.
- Level 5 (spoken, high): Strongly spontaneous; frequent fillers, discourse markers, turn-yielding cues; more repetition and repairs; facts preserved.

Annotation note
Do not penalize minor punctuation or casing differences unless they affect meaning or readability. Focus on conversational cues and overall spoken feel while ensuring all information remains intact.
"""

# ----------------------------------------------------------------------------
# (2) Few-shot addendum — per-level calibration + three pairwise demos.
# All texts are short real utterances drawn from the WSPairs training set.
# Their g(·) levels are the kmeans-cluster labels at construction time.
# ----------------------------------------------------------------------------
PER_LEVEL_EXAMPLES = [
    (0, "Are you experiencing muscle aches or generalized joint pain?"),
    (1, "you're welcome , but i will need your phone number for the cat ."),
    (2, "okay . um . which day and at what time does plan ."),
    (3, "Okay, well I'm sorry to hear that you're not feeling well. Um, the cough, when did that start?"),
    (4, "All right. Should be should be dug into this thing then."),
    (5, "OK, when the cough started uh were you bringing up, um were you bringing up sputum at that time?"),
]

# Three pairwise demos. Each tuple is (text_a, level_a, text_b, level_b, gold, reason).
# `gold` ∈ {"A","B"}; we deliberately mix A/B so the model can't shortcut.
PAIRWISE_DEMOS = [
    # Demo 1 — high margin, A is more spoken (level 5 vs 0).
    (
        "OK, when the cough started uh were you bringing up, um were you bringing up sputum at that time?",
        5,
        "When the cough started, were you bringing up sputum at that time?",
        0,
        "A",
        "A contains fillers (uh, um) and a self-repair repetition; B is a single formal question.",
    ),
    # Demo 2 — medium margin, B is more spoken (level 3 vs 0).
    (
        "Are you experiencing muscle aches or generalized joint pain?",
        0,
        "Okay, well I'm sorry to hear that you're not feeling well. Um, the cough, when did that start?",
        3,
        "B",
        "B has discourse markers (Okay, well, Um) and fragmentation; A is a formal medical question.",
    ),
    # Demo 3 — small margin (level 2 vs 1), A is more spoken.
    (
        "okay . um . which day and at what time does plan .",
        2,
        "you're welcome , but i will need your phone number for the cat .",
        1,
        "A",
        "A has two fillers (okay, um) and is more fragmented; B has only a contraction.",
    ),
]


def _render_demos() -> str:
    """Build the few-shot block (level calibration + pairwise demos)."""
    lines = []
    lines.append("Reference examples (one short utterance per g(·) level, for calibration only):")
    for lv, t in PER_LEVEL_EXAMPLES:
        lines.append(f"  Level {lv}: {t!r}")
    lines.append("")
    lines.append("Pairwise demonstrations (the JSON answer follows  =>):")
    for i, (ta, la, tb, lb, gold, reason) in enumerate(PAIRWISE_DEMOS, 1):
        lines.append("")
        lines.append(f"Demo {i}:")
        lines.append(f"Text A:")
        lines.append(ta)
        lines.append(f"Text B:")
        lines.append(tb)
        ans = '{"more_spoken": "%s", "reason": "%s"}' % (gold, reason)
        lines.append(f"=> {ans}")
    return "\n".join(lines)


_DEMO_BLOCK = _render_demos()


# ----------------------------------------------------------------------------
# (3) Full prompt builder.
# ----------------------------------------------------------------------------
def build_compare_prompt(text_a: str, text_b: str) -> str:
    """Render the full few-shot pairwise comparison prompt for one pair."""
    return (
        f"{GUIDELINES}\n"
        f"---\n"
        f"{_DEMO_BLOCK}\n\n"
        f"---\n"
        f"Now answer the following pair. Respond with a JSON object only, "
        f"on the same line as `=>`. No prose before or after the JSON. "
        f'Fields: "more_spoken" ∈ {{"A","B","Tie"}}, "reason" (<=20 words).\n\n'
        f"Text A:\n{text_a.strip()}\n"
        f"Text B:\n{text_b.strip()}\n"
        f"=> "
    )
