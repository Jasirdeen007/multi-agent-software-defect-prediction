from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class DebertaEmbeddingExtractor:
    """Extracts frozen sentence embeddings using DeBERTa-v3 with GPU/FP16 support."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 256,
        batch_size: int = 32,
        use_fp16: bool = True,
        pooling: Literal["cls", "mean"] = "cls",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling = pooling

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; refusing to fall back to CPU.")

        print(f"Loading DeBERTa model: {model_name} on {self.device} (FP16={use_fp16})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if self.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _collate_fn(self, batch_texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def extract_embeddings(self, texts: list[str], split_name: str = "dataset") -> np.ndarray:
        dataset = TextDataset(texts)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
        )

        all_embeddings: list[np.ndarray] = []
        desc = f"Extracting {split_name} embeddings"

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state

                if self.pooling == "cls":
                    batch_embeds = last_hidden_state[:, 0, :]
                else:
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeds = sum_embeddings / sum_mask

                all_embeddings.append(batch_embeds.cpu().float().numpy())

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        stacked = np.vstack(all_embeddings)
        print(f"Completed {split_name}: shape {stacked.shape}, dtype={stacked.dtype}")
        return stacked

    def extract_embeddings_to_memmap(
        self, texts: list[str], cache_path: str | Path, split_name: str = "dataset"
    ) -> np.ndarray:
        """Extract batches directly to disk so large splits do not need a RAM-sized copy."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_path = path.with_suffix(path.suffix + ".progress.json")
        hidden_size = int(self.model.config.hidden_size)
        signature = {
            "model": self.model_name,
            "rows": len(texts),
            "max_length": self.max_length,
            "pooling": self.pooling,
            "hidden_size": hidden_size,
        }

        done = 0
        if path.exists() and state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if all(state.get(key) == value for key, value in signature.items()):
                    if state.get("complete"):
                        return np.load(path, mmap_mode="r")
                    done = int(state.get("rows_done", 0))
                    matrix = np.load(path, mmap_mode="r+")
                else:
                    path.unlink()
                    state_path.unlink()
                    matrix = np.lib.format.open_memmap(
                        path, mode="w+", dtype=np.float32, shape=(len(texts), hidden_size)
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                matrix = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(len(texts), hidden_size)
                )
        else:
            path.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            matrix = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(len(texts), hidden_size)
            )

        state = {**signature, "rows_done": done, "complete": False}
        micro_batch = self.batch_size
        checkpoint_rows = max(micro_batch, micro_batch * 100)
        for start in range(done, len(texts), micro_batch):
            end = min(start + micro_batch, len(texts))
            batch = self._collate_fn(texts[start:end])
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                if self.pooling == "cls":
                    vectors = hidden[:, 0, :]
                else:
                    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            matrix[start:end] = vectors.cpu().float().numpy()
            if end % checkpoint_rows < micro_batch or end == len(texts):
                matrix.flush()
                state.update({"rows_done": end, "complete": False})
                state_path.write_text(json.dumps(state), encoding="utf-8")
                print(f"{split_name}: {end:,}/{len(texts):,}", flush=True)

        matrix.flush()
        state.update({"rows_done": len(texts), "complete": True})
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return np.load(path, mmap_mode="r")

    def extract_parquet_to_memmap(
        self,
        parquet_path: str | Path,
        text_column: str,
        cache_path: str | Path,
        split_name: str,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Stream one split's text from Parquet and checkpoint embeddings to disk."""
        source = Path(parquet_path)
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_path = path.with_suffix(path.suffix + ".progress.json")
        parquet = pq.ParquetFile(source)
        rows = parquet.metadata.num_rows
        hidden_size = int(self.model.config.hidden_size)
        signature = {
            "model": self.model_name,
            "rows": rows,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "hidden_size": hidden_size,
            "source_bytes": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
        }
        done = 0
        if not force_recompute and path.exists() and state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if all(state.get(key) == value for key, value in signature.items()):
                    if state.get("complete"):
                        return np.load(path, mmap_mode="r")
                    done = int(state.get("rows_done", 0))
                    matrix = np.load(path, mmap_mode="r+")
                else:
                    done = 0
                    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, hidden_size))
            except (OSError, ValueError, json.JSONDecodeError):
                done = 0
                matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, hidden_size))
        else:
            done = 0
            matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, hidden_size))

        state = {**signature, "rows_done": done, "complete": False}
        cursor = 0
        stream_batch_rows = max(256, self.batch_size)
        for record_batch in parquet.iter_batches(columns=[text_column], batch_size=stream_batch_rows):
            texts = record_batch.column(0).to_pylist()
            block_end = cursor + len(texts)
            if block_end <= done:
                cursor = block_end
                continue
            offset = max(0, done - cursor)
            for local_start in range(offset, len(texts), self.batch_size):
                local_end = min(local_start + self.batch_size, len(texts))
                start, end = cursor + local_start, cursor + local_end
                batch = self._collate_fn(texts[local_start:local_end])
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.inference_mode():
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden = output.last_hidden_state
                    if self.pooling == "cls":
                        vectors = hidden[:, 0, :]
                    else:
                        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                        vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                matrix[start:end] = vectors.cpu().float().numpy()
            cursor = block_end
            matrix.flush()
            state.update({"rows_done": cursor, "complete": False})
            state_path.write_text(json.dumps(state), encoding="utf-8")
            if cursor % 10_000 < stream_batch_rows or cursor == rows:
                print(f"{split_name}: {cursor:,}/{rows:,}", flush=True)

        if cursor != rows:
            raise RuntimeError(f"Parquet stream read {cursor} rows, expected {rows} for {split_name}.")
        matrix.flush()
        state.update({"rows_done": rows, "complete": True})
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return np.load(path, mmap_mode="r")


def get_or_extract_embeddings(
    extractor: DebertaEmbeddingExtractor,
    texts: list[str],
    cache_path: str | Path,
    split_name: str = "split",
    force_recompute: bool = False,
) -> np.ndarray:
    cache_file = Path(cache_path)
    state_file = cache_file.with_suffix(cache_file.suffix + ".progress.json")
    if cache_file.exists() and not force_recompute:
        try:
            embeddings = np.load(cache_file, mmap_mode="r")
            if len(embeddings) == len(texts):
                print(f"Loaded {len(embeddings):,} cached embeddings from: {cache_file}")
                return embeddings
            print(f"Cache size mismatch ({len(embeddings)} vs expected {len(texts)}); recomputing.")
        except (OSError, ValueError):
            print(f"Invalid embedding cache at {cache_file}; recomputing.")
    if force_recompute:
        cache_file.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)
    return extractor.extract_embeddings_to_memmap(texts, cache_file, split_name=split_name)


def get_or_extract_parquet_embeddings(
    extractor: DebertaEmbeddingExtractor,
    parquet_path: str | Path,
    text_column: str,
    cache_path: str | Path,
    split_name: str,
    force_recompute: bool = False,
) -> np.ndarray:
    cache_file = Path(cache_path)
    state_file = cache_file.with_suffix(cache_file.suffix + ".progress.json")
    if force_recompute:
        cache_file.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)
    elif cache_file.exists() and state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            source = Path(parquet_path)
            expected_rows = pq.ParquetFile(source).metadata.num_rows
            if (
                state.get("complete")
                and state.get("model") == extractor.model_name
                and state.get("rows") == expected_rows
                and state.get("max_length") == extractor.max_length
                and state.get("pooling") == extractor.pooling
                and state.get("source_bytes") == source.stat().st_size
                and state.get("source_mtime_ns") == source.stat().st_mtime_ns
            ):
                return np.load(cache_file, mmap_mode="r")
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return extractor.extract_parquet_to_memmap(
        parquet_path, text_column, cache_file, split_name, force_recompute=force_recompute
    )


if __name__ == "__main__":
    import argparse
    from src.baseline.data_splitter import load_dataset, split_dataset

    parser = argparse.ArgumentParser(description="Extract DeBERTa embeddings")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--dataset-path", type=str, default="bughubs_canonical.parquet")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--out-dir", type=str, default="data/embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    print(f"--- Loading dataset for embedding extraction ---")
    df = load_dataset(args.dataset_path, sample_size=args.sample_size, random_seed=42)
    train_df, val_df, test_df = split_dataset(df, random_seed=42)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extractor = DebertaEmbeddingExtractor(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=True,
    )

    get_or_extract_embeddings(
        extractor,
        train_df["model_text"].tolist(),
        cache_path=out_path / f"train_embeddings_{len(train_df)}.npy",
        split_name="Train",
        force_recompute=args.force_recompute,
    )
    get_or_extract_embeddings(
        extractor,
        val_df["model_text"].tolist(),
        cache_path=out_path / f"val_embeddings_{len(val_df)}.npy",
        split_name="Val",
        force_recompute=args.force_recompute,
    )
    get_or_extract_embeddings(
        extractor,
        test_df["model_text"].tolist(),
        cache_path=out_path / f"test_embeddings_{len(test_df)}.npy",
        split_name="Test",
        force_recompute=args.force_recompute,
    )
    print(f"All embeddings extracted and saved to {out_path}!")

