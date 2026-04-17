from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import re
from collections import Counter


# -------------------------
# Data structures
# -------------------------
@dataclass(frozen=True)
class TGQAEvent:
    story_id: str
    fact: str                 # text inside parentheses or raw fallback
    start_year: Optional[int] # parsed from "starts at YYYY"
    raw: str                  # original TG string


@dataclass(frozen=True)
class TGQACloze:
    story_id: str
    prompt: str
    gold: str
    start_year: Optional[int]
    raw_fact: str


# -------------------------
# Parsing TG strings
# -------------------------
YEAR_PAT = re.compile(r"starts at (\d{4})")
FACT_PAT = re.compile(r"\((.*?)\)")

def parse_tg_event(story_id: str, tg_string: str) -> TGQAEvent:
    year_m = YEAR_PAT.search(tg_string)
    year = int(year_m.group(1)) if year_m else None

    fact_m = FACT_PAT.search(tg_string)
    fact = fact_m.group(1) if fact_m else tg_string

    return TGQAEvent(story_id=story_id, fact=fact, start_year=year, raw=tg_string)


def build_event_list(dataset_split: List[Dict[str, Any]]) -> List[TGQAEvent]:
    events: List[TGQAEvent] = []
    for ex in dataset_split:
        sid = str(ex.get("id", ""))
        for t in (ex.get("TG") or []):
            events.append(parse_tg_event(sid, t))
    return events


# -------------------------
# Verbalization: fact -> cloze
# -------------------------
def extract_sro(fact: str) -> Optional[Tuple[str, str, str]]:
    """
    Heuristic subject/relation/object extraction:
      - subject = first 2 tokens (synthetic names)
      - object = last 1-3 tokens
      - relation = middle tokens
    """
    toks = fact.split()
    if len(toks) < 4:
        return None

    subject = " ".join(toks[:2])

    for obj_len in (3, 2, 1):
        if len(toks) - 2 - obj_len <= 0:
            continue
        obj = " ".join(toks[-obj_len:])
        rel = " ".join(toks[2:-obj_len]).strip()
        if rel:
            return subject, rel, obj

    return None


def build_cloze(event: TGQAEvent) -> Optional[TGQACloze]:
    parsed = extract_sro(event.fact)
    if parsed is None:
        return None
    subject, rel, obj = parsed

    if event.start_year is not None:
        prompt = f"In {event.start_year}, {subject} {rel} ____."
    else:
        # Untimed facts are still evaluable as generic cloze
        prompt = f"{subject} {rel} ____."

    return TGQACloze(
        story_id=event.story_id,
        prompt=prompt,
        gold=obj,
        start_year=event.start_year,
        raw_fact=event.fact,
    )


def build_cloze_set(events: List[TGQAEvent], require_year: bool = True) -> List[TGQACloze]:
    out: List[TGQACloze] = []
    for ev in events:
        if require_year and ev.start_year is None:
            continue
        c = build_cloze(ev)
        if c is not None:
            out.append(c)
    return out


# -------------------------
# Scoring (reuse same style)
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


# -------------------------
# Full eval loop
# -------------------------
def evaluate_tgqa(
    model_generate_fn: Callable[[str], str],
    cloze_items: List[TGQACloze],
    verbose: bool = False,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    if max_examples is not None:
        cloze_items = cloze_items[:max_examples]

    ems, cms, f1s = [], [], []

    for item in cloze_items:
        pred = model_generate_fn(item.prompt)

        em = exact_match(pred, item.gold)
        cm = contains_match(pred, item.gold)
        f1 = token_f1(pred, item.gold)

        ems.append(float(em))
        cms.append(float(cm))
        f1s.append(float(f1))

        if verbose and not cm:
            print("Story:", item.story_id, "| year:", item.start_year)
            print("Prompt:", item.prompt)
            print("Gold:", item.gold)
            print("Pred:", pred)
            print("-" * 80)

    return {
        "n": float(len(cloze_items)),
        "exact_match": float(sum(ems) / max(1, len(ems))),
        "contains": float(sum(cms) / max(1, len(cms))),
        "f1": float(sum(f1s) / max(1, len(f1s))),
    }

