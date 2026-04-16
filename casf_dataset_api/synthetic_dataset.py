"""SyntheticDataset — TemporalDataset implementation for the Gemini-generated
synthetic CASM dataset.

This class is the drop-in dataset replacement for TemporalWikiDataset when
cfg.dataset_name == "synthetic".  It reads from files produced by the data
preparation scripts:

    data/build_probes.py   -> data/probes.json          (probe objects)
    data/build_passages.py -> data/passages.json         (thin template passages)
    dataset_utils/generate_dataset.py -> data/augmented/synthetic/<period>.csv
                                                         (augmented passages)

Period names: "2018", "2020", "2022", "2024"

Period "2018" semantics
-----------------------
The initial period has no predecessor, so no fact counts as "changed".
"changed" split contains ALL facts (with is_changed=False) so the trainer
writes them to the MemoryRegistry as new slots.  "unchanged" is empty.

Augmented passages
------------------
When use_augmented=True, the dataset reads from data/augmented/synthetic/
using the period name directly as the filename (e.g. 2018.csv, 2020.csv).

Thin passages (default)
-----------------------
When use_augmented=False (the default), the dataset reads from
data/passages.json — single-sentence templates suitable for fast
iteration and pipeline verification.
"""
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .casf_types import Probe
from .dataset import TemporalDataset

VALID_PERIODS = ["2018", "2020", "2022", "2024"]
VALID_SPLITS = {"train", "changed", "unchanged"}


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROBES_PATH = _PROJECT_ROOT / "data" / "probes.json"
_DEFAULT_PASSAGES_PATH = _PROJECT_ROOT / "data" / "passages.json"
_DEFAULT_AUGMENTED_DIR = _PROJECT_ROOT / "data" / "augmented" / "synthetic"


def _deserialise_probe(d: dict) -> Probe:
    """Reconstruct a Probe from a dataclasses.asdict() dict."""
    return Probe(**d)


class SyntheticDataset(TemporalDataset):
    """File-backed synthetic dataset for one time period.

    Parameters
    ----------
    period:
        One of "2018", "2020", "2022", "2024".
    probes_path:
        Path to data/probes.json produced by build_probes.py.
    passages_path:
        Path to data/passages.json produced by build_passages.py.
    augmented_dir:
        Directory containing augmented CSV files from generate_dataset.py.
    use_augmented:
        If True, load passages from the augmented CSVs instead of the thin
        template passages.json.  Requires generate_dataset.py to have been
        run first.
    """

    def __init__(
        self,
        period: str,
        *,
        probes_path: Path | str = _DEFAULT_PROBES_PATH,
        passages_path: Path | str = _DEFAULT_PASSAGES_PATH,
        augmented_dir: Path | str = _DEFAULT_AUGMENTED_DIR,
        use_augmented: bool = False,
    ) -> None:
        if period not in VALID_PERIODS:
            raise ValueError(
                f"Unknown period {period!r}. Must be one of {VALID_PERIODS}"
            )
        self.period = period
        self.snapshot_id = period
        self._probes_path = Path(probes_path)
        self._passages_path = Path(passages_path)
        self._augmented_dir = Path(augmented_dir)
        self._use_augmented = use_augmented

        self._loaded_split: Optional[str] = None
        self._probes: dict[str, list[Probe]] = {}
        self._passages: list[str] = []

    # ------------------------------------------------------------------
    # TemporalDataset ABC

    def load(self, split: str) -> None:
        """Load the dataset for the given split.

        "train"     — load training passages
        "changed"   — load probes for facts that changed at this period boundary
        "unchanged" — load probes for facts that stayed the same
        """
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}. Must be one of {sorted(VALID_SPLITS)}"
            )
        self._loaded_split = split
        if split == "train":
            self._ensure_passages_loaded()
        else:
            self._ensure_probes_loaded(split)

    def get_probes(self, split: Optional[str] = None) -> list[Probe]:
        """Return Probe objects for the requested split.

        Raises ValueError if no split has been loaded and none is specified.
        """
        target = split or self._loaded_split
        if target is None or target == "train":
            raise ValueError(
                "No probe split loaded. Call load('changed') or load('unchanged') first."
            )
        if target not in self._probes:
            self._ensure_probes_loaded(target)
        return self._probes[target]

    def get_train_passages(self) -> list[str]:
        """Return raw text passages for fine-tuning (list[str])."""
        if not self._passages:
            self._ensure_passages_loaded()
        return self._passages

    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        """Return (old_probe, new_probe) pairs for facts that changed this period.

        A "pair" is (unchanged_probe_from_prev_period, changed_probe_this_period).
        Since ContradictionDetector sets previous_value in-place, freshly loaded
        probes will have previous_value=None and is_contradiction=False until the
        detector runs.  This method instead uses is_changed as the ground-truth
        signal to build pairs from the current split data.
        """
        self._ensure_probes_loaded("changed")
        self._ensure_probes_loaded("unchanged")

        unchanged_index = {
            (p.subject, p.relation): p for p in self._probes.get("unchanged", [])
        }
        pairs: list[tuple[Probe, Probe]] = []
        for probe in self._probes.get("changed", []):
            if probe.is_changed:
                old = unchanged_index.get((probe.subject, probe.relation))
                if old is not None:
                    pairs.append((old, probe))
        return pairs

    # ------------------------------------------------------------------
    # Internal loading helpers

    def _ensure_probes_loaded(self, split: str) -> None:
        if split in self._probes:
            return
        if not self._probes_path.exists():
            raise FileNotFoundError(
                f"probes.json not found at {self._probes_path}. "
                "Run: uv run python data/build_probes.py"
            )
        raw = json.loads(self._probes_path.read_text())
        period_data = raw.get(self.period, {})
        for s in ("changed", "unchanged"):
            self._probes[s] = [
                _deserialise_probe(d) for d in period_data.get(s, [])
            ]

    def _ensure_passages_loaded(self) -> None:
        if self._passages:
            return
        if self._use_augmented:
            self._passages = self._load_augmented_passages()
        else:
            self._passages = self._load_thin_passages()

    def _load_thin_passages(self) -> list[str]:
        if not self._passages_path.exists():
            raise FileNotFoundError(
                f"passages.json not found at {self._passages_path}. "
                "Run: uv run python data/build_passages.py"
            )
        raw = json.loads(self._passages_path.read_text())
        passages = raw.get(self.period, [])
        # Guarantee list[str]
        return [str(p) for p in passages]

    def _load_augmented_passages(self) -> list[str]:
        csv_path = self._augmented_dir / f"{self.period}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Augmented CSV not found at {csv_path}. "
                "Run: uv run python dataset_utils/generate_dataset.py "
                f"--prompts-dir dataset_utils/prompts/synthetic "
                f"--outdir {self._augmented_dir}"
            )
        passages: list[str] = []
        prefix = f"[{self.period}] "
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("text") or "").strip()
                if text and text.upper() != "ERROR":
                    passages.append(prefix + text)
        return passages
