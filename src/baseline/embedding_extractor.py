from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
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
        adapter_path: str | Path | None = None,
        max_length: int = 256,
        batch_size: int = 32,
        use_fp16: bool = True,
        pooling: Literal["cls", "mean"] = "cls",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling = pooling

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Check if model_name itself is an adapter directory or if adapter_path is provided
        eff_adapter = None
        if adapter_path and Path(adapter_path).exists():
            eff_adapter = Path(adapter_path)
        elif Path(model_name).is_dir() and (Path(model_name) / "adapter_config.json").exists():
            eff_adapter = Path(model_name)

        if eff_adapter is not None:
            import json
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification

            with open(eff_adapter / "adapter_config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model_name = cfg.get("base_model_name_or_path", "microsoft/deberta-v3-small")
            print(f"Loading base model '{base_model_name}' with LoRA adapter from '{eff_adapter}' on {self.device} (FP16={use_fp16})...")
            self.tokenizer = AutoTokenizer.from_pretrained(eff_adapter)
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
            peft_model = PeftModel.from_pretrained(base_model, str(eff_adapter))
            self.model = peft_model.base_model.model.deberta
        else:
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


def get_or_extract_embeddings(
    extractor: DebertaEmbeddingExtractor,
    texts: list[str],
    cache_path: str | Path,
    split_name: str = "split",
    force_recompute: bool = False,
) -> np.ndarray:
    cache_file = Path(cache_path)
    if cache_file.exists() and not force_recompute:
        print(f"Found cached embeddings for {split_name} at: {cache_file}")
        embeddings = np.load(cache_file)
        if len(embeddings) == len(texts):
            print(f"Loaded {len(embeddings):,} embeddings from cache (dim={embeddings.shape[1]}).")
            return embeddings
        else:
            print(f"Cache size mismatch ({len(embeddings)} vs expected {len(texts)}). Recomputing...")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    embeddings = extractor.extract_embeddings(texts, split_name=split_name)
    np.save(cache_file, embeddings)
    print(f"Saved {split_name} embeddings to: {cache_file}")
    return embeddings


if __name__ == "__main__":
    import argparse
    from src.baseline.data_splitter import load_dataset, split_dataset

    parser = argparse.ArgumentParser(description="Extract DeBERTa embeddings (supports base or LoRA models)")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--dataset-path", type=str, default="bughubs_canonical.parquet")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--out-dir", type=str, default="data/embeddings/finetuned")
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
        adapter_path=args.adapter_dir,
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

