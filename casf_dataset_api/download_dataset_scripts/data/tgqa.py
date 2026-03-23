import re
from typing import Optional, Dict, Tuple
from datasets import load_dataset
from ...casf_types import Probe
from ...dataset import TemporalDataset

# Relations that are typically "single-valued" (mutually exclusive),
# so a new value likely supersedes the old one.
# This is heuristic because TGQA relations are free-form strings.
_EXCLUSIVE_REL_HINTS = [
    "was born", "born in", "was born in",         # birthplace
    "died", "died in",              # death place/time
    "was married", "married to",    # spouse
    "is ceo", "ceo of",             # role
    "served as", "was president", "was prime minister",
    "capital of", "located in",     # location-type relations
]

def _is_exclusive_relation(relation: str) -> bool:
    r = (relation or "").lower().strip()
    return any(h in r for h in _EXCLUSIVE_REL_HINTS)
VALID_SPLITS = ["train", "val", "test"]

YEAR_PAT = re.compile(r"starts at (\d{4})")
FACT_PAT = re.compile(r"\((.*?)\)")


def _parse_tg(tg: str) -> tuple[str, Optional[str]]:
    ym = YEAR_PAT.search(tg)
    year = ym.group(1) if ym else None
    fm = FACT_PAT.search(tg)
    fact = fm.group(1) if fm else tg
    return fact, year


def _extract_sro(fact: str) -> Optional[tuple[str, str, str]]:
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


def _cloze(subject: str, relation: str, year: Optional[str]) -> str:
    if year:
        return f"In {year}, {subject} {relation} ____."
    return f"{subject} {relation} ____."


class TGQADataset(TemporalDataset):
    """
    TGQA wrapper that converts TG events into cloze probes.
    """

    def __init__(self, config: str = "TGQA_Story_TG_Trans", require_year: bool = True):
        self.config = config
        self.require_year = require_year
        self.snapshot_id = None
        self._ds = None
        self._loaded_split: Optional[str] = None
        self._probes: dict[str, list[Probe]] = {}
        self._passages: dict[str, list[str]] = {}

    def load(self, split: str) -> None:
        if split == "validation":
            split = "val"
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Must be one of {VALID_SPLITS}")
        self._loaded_split = split

        if self._ds is None:
            self._ds = load_dataset("sxiong/TGQA", self.config)

    def _load_probes(self, split: str) -> None:
        assert self._ds is not None
        probes: list[Probe] = []

        # track last object in-story for (story_id, subject, relation)
        last_obj: Dict[Tuple[str, str, str], str] = {}

        for row in self._ds[split]:
            story_id = str(row.get("id", ""))
            for tg in (row.get("TG") or []):
                fact, year = _parse_tg(tg)
                if self.require_year and not year:
                    continue

                sro = _extract_sro(fact)
                if not sro:
                    continue
                subject, relation, obj = sro

                key = (story_id, subject, relation)
                prev = last_obj.get(key)
                last_obj[key] = obj

                # Only treat updates as contradictions for "exclusive" relations.
                if _is_exclusive_relation(relation) and (prev is not None) and (prev != obj):
                    prev_val = prev
                    is_changed = True
                else:
                    prev_val = None
                    is_changed = False
                

                probes.append(
                    Probe(
                        prompt=_cloze(subject, relation, year),
                        ground_truth=obj,
                        relation=relation,
                        subject=subject,
                        current_value=obj,
                        source="tgqa",
                        is_changed=is_changed,
                        timestamp=year,
                        previous_value=prev_val,
                        valid_from=year,
                        valid_until=None,
                        metadata={
                            "story_id": story_id,
                            "raw_tg": tg,
                            "raw_fact": fact,
                            "event_year": year,
                        },
                    )
                )

        self._probes[split] = probes

    def _load_passages(self, split: str) -> None:
        assert self._ds is not None
        self._passages[split] = [str(row.get("story", "")) for row in self._ds[split]]

    def get_probes(self, split: Optional[str] = None) -> list[Probe]:
        target = split or self._loaded_split
        if target is None:
            raise ValueError("No split loaded. Call load('train'|'val'|'test') first.")
        if target not in self._probes:
            self._load_probes(target)
        return self._probes[target]

    def get_train_passages(self) -> list[str]:
        if self._loaded_split != "train":
            raise ValueError("Call load('train') before get_train_passages().")
        if "train" not in self._passages:
            self._load_passages("train")
        return self._passages["train"]

    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        # For TGQA we can expose within-story updates as contradiction pairs.
        probes = self.get_probes()
        pairs: list[tuple[Probe, Probe]] = []

        last_seen: Dict[Tuple[str, str, str], Probe] = {}
        for p in probes:
            story_id = p.metadata.get("story_id", "")
            key = (story_id, p.subject, p.relation)
            prev = last_seen.get(key)
            if _is_exclusive_relation(p.relation) and prev and prev.current_value != p.current_value:
                pairs.append((prev, p))
            last_seen[key] = p

        return pairs

if __name__ == "__main__":
    ds = TGQADataset()
    ds.load("train")
    probes = ds.get_probes()

    print("Loaded probes:", len(probes))
    print("First probe:", probes[0])

    # Optional quick stats
    n_changed = sum(1 for p in probes if p.is_changed)
    n_contra = sum(1 for p in probes if p.is_contradiction)
    print("Changed probes:", n_changed)
    print("Contradiction probes:", n_contra)

    pairs = ds.get_contradiction_pairs()
    print("Contradiction pairs:", len(pairs))
    print("Sample contradiction pairs:")
    for old, new in pairs[:5]:
        print(f"{old.subject} | {old.relation} | {old.current_value} -> {new.current_value} (year {new.timestamp})")
