import pytest
import torch
from torch.utils.data import Dataset

from torch_uncertainty.datamodules.classification.sst2 import Sst2DataModule


class DummySplit:
    def __init__(self, rows):
        """HF-like split that supports map()/set_format() and indexing."""
        self._rows = list(rows)
        self._keep_columns = None

    @property
    def column_names(self):
        """Return all column names present in the split."""
        names = set()
        for r in self._rows:
            names |= set(r.keys())
        return list(names)

    def map(self, fn, batched=True, remove_columns=()):
        """Apply a batched mapping function and drop columns."""
        if not batched:
            raise NotImplementedError("Only batched=True supported in stub")

        batch = {name: [r.get(name) for r in self._rows] for name in self.column_names}

        new_cols = fn(batch)  # dict of lists

        keep = {k: batch[k] for k in batch if k not in set(remove_columns)}
        keep.update(new_cols)

        n = len(next(iter(keep.values()))) if keep else len(self._rows)
        new_rows = [{k: keep[k][i] for k in keep} for i in range(n)]
        return DummySplit(new_rows)

    def set_format(self, fmt="torch", columns=None, **kwargs):
        """Mimic HF set_format: keep only requested columns on __getitem__.

        Accepts both fmt=... and type=... (HF uses type=).
        """
        if "type" in kwargs and fmt == "torch":
            fmt = kwargs["type"]
        self._keep_columns = list(columns) if columns is not None else None
        return self

    def __len__(self):
        """Number of rows in the split."""
        return len(self._rows)

    def __getitem__(self, idx):
        """Get a row (optionally filtered to requested columns)."""
        r = self._rows[idx]
        if self._keep_columns is not None:
            r = {k: r[k] for k in self._keep_columns if k in r}
        return r


class DummyDatasetDict(dict):
    """HF-like container with split keys."""

    def map(self, fn, batched=True, remove_columns=()):
        """Apply map to each contained split."""
        return DummyDatasetDict(
            {k: v.map(fn, batched=batched, remove_columns=remove_columns) for k, v in self.items()}
        )

    def set_format(self, fmt="torch", columns=None, **kwargs):
        """Apply set_format to each contained split.

        Accepts both fmt=... and type=... (HF uses type=).
        """
        if "type" in kwargs and fmt == "torch":
            fmt = kwargs["type"]
        for v in self.values():
            v.set_format(fmt=fmt, columns=columns)
        return self


class DummyTokenizer:
    """Minimal tokenizer stub returning fixed-length ids and masks."""

    def __init__(self, max_id=1000):
        self.max_id = max_id

    def __call__(self, *args, max_length=128, truncation=True, padding="max_length"):
        if len(args) == 1:
            texts = args[0]
        elif len(args) == 2:
            t1, t2 = args
            texts = [f"{a} {b}" for a, b in zip(t1, t2, strict=False)]
        else:
            raise ValueError("Unexpected tokenizer inputs")

        n = len(texts)
        out_ids, out_mask = [], []
        for i in range(n):
            seq_len = min(16, max_length)
            ids = [(i + j) % self.max_id for j in range(seq_len)]
            mask = [1] * seq_len
            if seq_len < max_length:
                pad = max_length - seq_len
                ids += [0] * pad
                mask += [0] * pad
            out_ids.append(ids)
            out_mask.append(mask)
        return {"input_ids": out_ids, "attention_mask": out_mask}

    @classmethod
    def from_pretrained(cls, *_, **__):
        return cls()


@pytest.fixture
def patch_hf(monkeypatch):
    """Patch tokenizer and load_dataset in the module where Sst2DataModule lives."""
    monkeypatch.setattr(
        f"{Sst2DataModule.__module__}.AutoTokenizer",
        DummyTokenizer,
        raising=True,
    )

    def _fake_load_dataset(path, name=None, split=None, download_config=None):  # noqa: ARG001
        if path == "glue" and name == "sst2":
            train = DummySplit(
                [
                    {"sentence": "great movie", "label": 1, "idx": 0},
                    {"sentence": "bad film", "label": 0, "idx": 1},
                    {"sentence": "okay", "label": 1, "idx": 2},
                ]
            )
            val = DummySplit(
                [
                    {"sentence": "not good", "label": 0, "idx": 3},
                    {"sentence": "wonderful", "label": 1, "idx": 4},
                ]
            )
            if split is None:
                return DummyDatasetDict(train=train, validation=val)
            return {"train": train, "validation": val}[split]

        # Near OOD
        if path == "yelp_polarity":
            return DummySplit([{"text": "food was amazing", "label": 1}])
        if path == "amazon_polarity":
            return DummySplit([{"content": "terrible product", "label": 0}])

        # Far OOD
        if path == "ag_news":
            return DummySplit([{"text": "stocks fell today", "label": 2}])
        if path == "SetFit/20_newsgroups":
            return DummySplit([{"text": "comp.graphics topic", "label": 5}])
        if path == "SetFit/TREC-QC":
            return DummySplit([{"text": "what is AI?", "label": 0}])
        if path == "glue" and name == "mnli":
            return DummySplit([{"premise": "cats sleep", "hypothesis": "animals rest", "label": 1}])
        if path == "glue" and name == "rte":
            return DummySplit(
                [{"sentence1": "A man runs", "sentence2": "A person jogs", "label": 1}]
            )
        if path == "wmt16" and name == "ro-en":
            return DummySplit([{"translation": {"ro": "salut", "en": "hello"}, "label": 0}])

        raise KeyError(f"Unhandled dataset: {(path, name, split)}")

    monkeypatch.setattr(
        f"{Sst2DataModule.__module__}.load_dataset",
        _fake_load_dataset,
        raising=True,
    )


class TestSst2DataModule:
    def test_id_only(self, patch_hf):
        dm = Sst2DataModule(
            model_name="bert-base-uncased",
            max_len=32,
            batch_size=4,
            local_files_only=True,
            eval_ood=False,
            num_workers=0,
            persistent_workers=False,
        )

        dm.prepare_data()
        dm.setup("fit")
        dm.setup("test")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loaders = dm.test_dataloader()

        assert isinstance(dm.test, Dataset)
        assert len(test_loaders) == 1  # only ID

        xb, yb = next(iter(train_loader))
        assert {"input_ids", "attention_mask"} <= set(xb.keys())
        assert yb.dtype == torch.long

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == []
        assert idx["val_ood"] == []
        assert idx["near_oods"] == []
        assert idx["far_oods"] == []
        assert idx["shift"] == []

        _ = next(iter(val_loader))
        _ = next(iter(test_loaders[0]))

    def test_with_ood(self, patch_hf):
        dm = Sst2DataModule(
            model_name="bert-base-uncased",
            max_len=32,
            batch_size=4,
            local_files_only=True,
            eval_ood=True,
            num_workers=0,
            persistent_workers=False,
        )

        dm.prepare_data()
        dm.setup("fit")
        dm.setup("test")

        assert isinstance(dm.test, Dataset)
        assert len(dm.near_oods) == 2  # yelp + amazon
        assert len(dm.far_oods) == 6  # ag_news, 20newsg, trec_qc, mnli_mm, rte, wmt16_en

        for ds in [*dm.near_oods, *dm.far_oods]:
            assert isinstance(ds, Dataset)
            assert hasattr(ds, "dataset_name")
            assert isinstance(ds.dataset_name, str)
            assert ds.dataset_name

        loaders = dm.test_dataloader()
        expected = 1 + 1 + len(dm.near_oods) + len(dm.far_oods)
        assert len(loaders) == expected

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == [1]
        assert idx["val_ood"] == []
        assert idx["near_oods"] == list(range(2, 2 + len(dm.near_oods)))
        assert idx["far_oods"] == list(
            range(2 + len(dm.near_oods), 2 + len(dm.near_oods) + len(dm.far_oods))
        )
        assert idx["shift"] == []

        (xb, yb) = next(iter(loaders[2]))
        assert {"input_ids", "attention_mask"} <= set(xb.keys())
        assert yb.dtype == torch.long
