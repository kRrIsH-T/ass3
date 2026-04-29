from __future__ import annotations

import csv
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig

AG_NEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, tokens: Sequence[str]):
        if list(tokens[:2]) != [self.PAD, self.UNK]:
            raise ValueError("Vocabulary must begin with <pad>, <unk>")
        self.itos = list(tokens)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int, max_size: int) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        # Alphabetical tie-breaking makes the vocabulary reproducible across platforms.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        words = [word for word, count in ranked if count >= min_frequency]
        return cls([cls.PAD, cls.UNK, *words[: max(0, max_size - 2)]])

    def encode(self, text: str, max_length: int) -> tuple[list[int], int]:
        ids = [self.stoi.get(token, 1) for token in tokenize(text)[:max_length]]
        length = max(1, len(ids))
        ids = ids or [self.stoi[self.UNK]]
        return ids + [0] * (max_length - len(ids)), length

    def __len__(self) -> int:
        return len(self.itos)


class EncodedNewsDataset(Dataset):
    def __init__(self, rows: Sequence[tuple[int, str]], vocabulary: Vocabulary, max_length: int):
        self.rows = list(rows)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label, text = self.rows[index]
        ids, length = self.vocabulary.encode(text, self.max_length)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


@dataclass(frozen=True)
class DataInfo:
    source: str
    train_samples: int
    validation_samples: int
    test_samples: int
    vocabulary_size: int
    num_classes: int = 4


def _tiny_rows() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    examples = {
        0: ["leaders meet for peace talks", "election results announced by government", "international summit opens"],
        1: ["team wins championship final", "striker scores twice in league game", "tennis player reaches final"],
        2: ["company reports higher quarterly profit", "markets rise after earnings", "bank announces new investment"],
        3: ["researchers reveal new space telescope", "software update fixes security issue", "chip maker launches processor"],
    }
    train = [(label, text) for label, texts in examples.items() for text in texts[:2]]
    test = [(label, texts[2]) for label, texts in examples.items()]
    return train, test


def _read_csv(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            rows.append((int(row[0]) - 1, f"{row[1]} {row[2]}".strip()))
    return rows


def _stratified_split(rows: Sequence[tuple[int, str]], validation_fraction: float, seed: int):
    groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)
    rng = random.Random(seed)
    train, validation = [], []
    for label in sorted(groups):
        group = groups[label][:]
        rng.shuffle(group)
        count = max(1, int(round(len(group) * validation_fraction)))
        if len(group) > 1:
            count = min(count, len(group) - 1)
        validation.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _balanced_limit(rows: Sequence[tuple[int, str]], limit: int | None, seed: int):
    if limit is None or len(rows) <= limit:
        return list(rows)
    groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = [row for i in range(limit) for row in groups[i % len(groups)][i // len(groups) : i // len(groups) + 1]]
    rng.shuffle(selected)
    return selected


class AGNewsDataModule:
    """One deterministic dataset/vocabulary shared by every compared model."""

    def __init__(self, config: DataConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.root = Path(config.output_dir) / "data" / "ag_news"
        self.vocabulary: Vocabulary | None = None
        self.datasets: dict[str, EncodedNewsDataset] = {}
        self.info: DataInfo | None = None

    def _download(self, split: str) -> Path:
        path = self.root / "raw" / f"{split}.csv"
        if path.exists() and path.stat().st_size > 0:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        request = urllib.request.Request(AG_NEWS_URLS[split], headers={"User-Agent": "sequence-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(path)
        return path

    def prepare(self) -> DataInfo:
        source = "AG News"
        if self.config.force_fallback:
            train_rows, test_rows = _tiny_rows()
            source = "built-in smoke fallback"
        else:
            try:
                train_rows = _read_csv(self._download("train"))
                test_rows = _read_csv(self._download("test"))
                if not train_rows or not test_rows:
                    raise ValueError("downloaded AG News files were empty")
            except (OSError, ValueError) as exc:
                if not self.config.offline_fallback:
                    raise RuntimeError("Unable to prepare AG News") from exc
                train_rows, test_rows = _tiny_rows()
                source = f"built-in smoke fallback ({type(exc).__name__})"

        train_rows, validation_rows = _stratified_split(
            train_rows, self.config.validation_fraction, self.seed
        )
        train_rows = _balanced_limit(train_rows, self.config.max_train_samples, self.seed)
        validation_rows = _balanced_limit(
            validation_rows, self.config.max_validation_samples, self.seed + 1
        )
        test_rows = _balanced_limit(test_rows, self.config.max_test_samples, self.seed + 2)
        vocabulary = Vocabulary.build(
            (text for _, text in train_rows), self.config.min_frequency, self.config.max_vocab_size
        )
        self.vocabulary = vocabulary
        self.datasets = {
            "train": EncodedNewsDataset(train_rows, vocabulary, self.config.max_length),
            "validation": EncodedNewsDataset(validation_rows, vocabulary, self.config.max_length),
            "test": EncodedNewsDataset(test_rows, vocabulary, self.config.max_length),
        }
        self.info = DataInfo(source, len(train_rows), len(validation_rows), len(test_rows), len(vocabulary))
        processed = self.root / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "metadata.json").write_text(
            json.dumps({**asdict(self.info), "seed": self.seed, "config": asdict(self.config)}, indent=2),
            encoding="utf-8",
        )
        (processed / "vocabulary.json").write_text(json.dumps(vocabulary.itos), encoding="utf-8")
        return self.info

    def loaders(self) -> dict[str, DataLoader]:
        if not self.datasets:
            self.prepare()
        common = dict(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        generator = torch.Generator().manual_seed(self.seed)
        return {
            "train": DataLoader(self.datasets["train"], shuffle=True, generator=generator, **common),
            "validation": DataLoader(self.datasets["validation"], shuffle=False, **common),
            "test": DataLoader(self.datasets["test"], shuffle=False, **common),
        }
