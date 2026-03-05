import io
import zipfile
import pandas as pd
from typing import Optional
from ..casf_types import Probe
from ..dataset import TemporalDataset
from ..verbalizer import Verbalizer

VALID_PERIODS = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
VALID_SPLITS = ["train", "changed", "unchanged"]

# Map period strings to zip file name fragments
PERIOD_TO_PROBES = {
    "aug_sep": "0801-0901",
    "sep_oct": "0901-1001",
    "oct_nov": "1001-1101",
    "nov_dec": "1101-1201",
}

PERIOD_TO_DIFFSET = {
    "aug_sep": "wikipedia_0809_gpt2.csv",
    "sep_oct": "wikipedia_0910_gpt2.csv",
    "oct_nov": "wikipedia_1011_gpt2.csv",
    "nov_dec": "wikipedia_1112_gpt2.csv",
}

PERIOD_BOUNDARIES = {
    "aug_sep":  ("2021-08", "2021-09"),
    "sep_oct":  ("2021-09", "2021-10"),
    "oct_nov":  ("2021-10", "2021-11"),
    "nov_dec":  ("2021-11", "2021-12"),
}

PERIOD_ORDER = VALID_PERIODS

# Paths to the downloaded zips — relative to project root
PROBES_ZIP   = "casf/download_dataset_scripts/data/TWiki_Probes.zip"
DIFFSETS_ZIP = "casf/download_dataset_scripts/data/TWiki_Diffsets.zip"


class TemporalWikiDataset(TemporalDataset):

    def __init__(self, period: str):
        if period not in VALID_PERIODS:
            raise ValueError(f"Unknown period {period!r}. Must be one of {VALID_PERIODS}")
        self.period = period
        self.snapshot_id = period
        self._verbalizer = Verbalizer()
        self._probes: dict[str, list[Probe]] = {}
        self._passages: list[str] = []
        self._loaded_split: Optional[str] = None

    def load(self, split: str) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Must be one of {VALID_SPLITS}")
        self._loaded_split = split
        if split == "train":
            self._load_passages()
        else:
            self._load_probes(split)

    def _load_passages(self) -> None:
        """Load raw training passages from local diffsets zip."""
        filename = PERIOD_TO_DIFFSET[self.period]
        with zipfile.ZipFile(DIFFSETS_ZIP, "r") as z:
            with z.open(f"TWiki_Diffsets/{filename}") as f:
                df = pd.read_csv(io.BytesIO(f.read()))

        # Return raw passages — PassageFilter owns dedup + stub removal
        text_col = self._detect_text_column(df)
        self._passages = [
            str(row).strip()
            for row in df[text_col]
            if str(row).strip()
        ]

    def _load_probes(self, split: str) -> None:
        """Load cloze probes from local probes zip."""
        is_changed = split == "changed"
        valid_from, valid_until = PERIOD_BOUNDARIES[self.period]
        period_prefix = PERIOD_TO_PROBES[self.period]
        filename = f"twiki_probes/{period_prefix}_{split}.csv"

        with zipfile.ZipFile(PROBES_ZIP, "r") as z:
            with z.open(filename) as f:
                df = pd.read_csv(io.BytesIO(f.read()))

        print(f"  Columns in {filename}: {list(df.columns)}")
        print(f"  Sample row:\n{df.iloc[0]}\n")

        probes = []
        for _, row in df.iterrows():
            subject  = str(row.get("subject", "") or "")
            relation = str(row.get("relation", "") or "")
            obj      = str(row.get("object", row.get("current_value", "")) or "")
            prev_obj = row.get("previous_value", row.get("prev_object", None))
            if pd.isna(prev_obj):
                prev_obj = None

            prompt = self._verbalizer.verbalize(subject, relation)
            if prompt is None:
                continue

            probe = Probe(
                prompt=prompt,
                ground_truth=obj,
                relation=relation,
                subject=subject,
                current_value=obj,
                source="temporalwiki",
                is_changed=is_changed,
                timestamp=self.period,
                previous_value=str(prev_obj) if prev_obj else None,
                valid_from=valid_from,
                valid_until=valid_until if is_changed else None,
                metadata={"period": self.period},
            )
            probes.append(probe)

        self._probes[split] = probes

    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Find the text column in the diffset CSV."""
        for candidate in ["text", "passage", "content", "sentence"]:
            if candidate in df.columns:
                return candidate
        # Fall back to the first string column
        for col in df.columns:
            if df[col].dtype == object:
                return col
        raise ValueError(f"Could not find a text column. Columns: {list(df.columns)}")

    def get_probes(self, split: Optional[str] = None) -> list[Probe]:
        target = split or self._loaded_split
        if target is None or target == "train":
            raise ValueError("No probe split loaded. Call load('changed') or load('unchanged') first.")
        if target not in self._probes:
            self._load_probes(target)
        return self._probes[target]

    def get_train_passages(self) -> list[str]:
        if not self._passages:
            self._load_passages()
        return self._passages

    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        changed   = self.get_probes("changed")
        unchanged = self.get_probes("unchanged")
        unchanged_index = {(p.subject, p.relation): p for p in unchanged}
        pairs = []
        for new_probe in changed:
            if new_probe.is_contradiction:
                old_probe = unchanged_index.get((new_probe.subject, new_probe.relation))
                if old_probe:
                    pairs.append((old_probe, new_probe))
        return pairs