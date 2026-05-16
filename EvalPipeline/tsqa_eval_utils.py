from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import re
from collections import Counter


# -------------------------
# Data structure
# -------------------------
@dataclass(frozen=True)
class TSQAExample:
    ex_id: str
    question: str
    context: str
    answers: List[str]
    source: str
    is_hard_negative: bool
    question_ts: Optional[datetime]
    evidence_ts: Optional[datetime]
    gap_days: Optional[int]
    question_type: str
    has_critical_dimensions: bool


# -------------------------
# Parsing helpers
# -------------------------
def parse_iso_ts(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return datetime.fromisoformat(s)


def compute_gap_days(qts: Optional[datetime], ets: Optional[datetime]) -> Optional[int]:
    if qts is None or ets is None:
        return None
    return (qts - ets).days


def parse_dimensions(dim_str: Any) -> Tuple[str, bool]:
    """dimensions is a JSON string; returns (question_type, has_critical_dimensions)."""
    if dim_str is None:
        return ("UNK", False)
    try:
        obj = json.loads(dim_str)
        return (obj.get("question_type", "UNK"), bool(obj.get("has_critical_dimensions", False)))
    except Exception:
        return ("UNK", False)


def to_example(raw: Dict[str, Any]) -> TSQAExample:
    qts = parse_iso_ts(raw.get("question_timestamp"))
    ets = parse_iso_ts(raw.get("evidence_timestamp"))
    gap = compute_gap_days(qts, ets)
    qtype, crit = parse_dimensions(raw.get("dimensions"))

    return TSQAExample(
        ex_id=str(raw.get("id", "")),
        question=str(raw.get("question", "")),
        context=str(raw.get("context", "")),
        answers=list(raw.get("answers") or []),
        source=str(raw.get("source", "")),
        is_hard_negative=bool(raw.get("is_hard_negative", False)),
        question_ts=qts,
        evidence_ts=ets,
        gap_days=gap,
        question_type=qtype,
        has_critical_dimensions=crit,
    )


def split_clean_vs_perturbed(examples: List[TSQAExample]) -> Tuple[List[TSQAExample], List[TSQAExample]]:
    clean, pert = [], []
    for ex in examples:
        (pert if ex.is_hard_negative else clean).append(ex)
    return clean, pert


# -------------------------
# Prompting (baseline)
# -------------------------
def build_prompt(ex: TSQAExample, include_context: bool = True) -> str:
    """Simple QA prompt; later you can prepend memory snippets here."""
    if include_context:
        return f"Question: {ex.question}\n\nContext:\n{ex.context}\n\nAnswer:"
    return f"Question: {ex.question}\n\nAnswer:"


# -------------------------
# Scoring (mirrors teammate)
# -------------------------
_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _punct.sub("", text)
    text = _ws.sub(" ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize(prediction) == normalize(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    return normalize(ground_truth) in normalize(prediction)


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def best_of_gold(prediction: str, golds: List[str]) -> Tuple[bool, bool, float]:
    """Return (best_exact, best_contains, best_f1) across all gold answers."""
    if not golds:
        return (False, False, 0.0)
    em = any(exact_match(prediction, g) for g in golds)
    cm = any(contains_match(prediction, g) for g in golds)
    f1 = max(token_f1(prediction, g) for g in golds)
    return (em, cm, f1)


# -------------------------
# Full eval loop (like teammate)
# -------------------------
def evaluate_tsqa(
    model_generate_fn: Callable[[str], str],
    examples: List[TSQAExample],
    include_context: bool = True,
    verbose: bool = False,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on TS-QA examples.
    Returns EM / contains / F1 averages.
    """
    if max_examples is not None:
        examples = examples[:max_examples]

    ems, cms, f1s = [], [], []

    for ex in examples:
        prompt = build_prompt(ex, include_context=include_context)
        pred = model_generate_fn(prompt)

        em, cm, f1 = best_of_gold(pred, ex.answers)
        ems.append(float(em))
        cms.append(float(cm))
        f1s.append(float(f1))

        if verbose and not cm:
            print("ID:", ex.ex_id, "| source:", ex.source, "| hard_neg:", ex.is_hard_negative)
            print("Q:", ex.question)
            print("Gold:", ex.answers)
            print("Pred:", pred)
            print("-" * 80)

    return {
        "n": float(len(examples)),
        "exact_match": float(sum(ems) / max(1, len(ems))),
        "contains": float(sum(cms) / max(1, len(cms))),
        "f1": float(sum(f1s) / max(1, len(f1s))),
    }

def bucket_gap_days(gap: Optional[int]) -> str:
    """Bucket timestamp gaps (in days) for slice analysis."""
    if gap is None:
        return "missing"
    if gap <= 7:
        return "0-7d"
    if gap <= 30:
        return "8-30d"
    if gap <= 365:
        return "31-365d"
    return "365d+"

