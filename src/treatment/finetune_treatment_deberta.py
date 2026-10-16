from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model


class DefectTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def finetune_treatment_deberta(
    dataset_path: str = "data/processed/treatment_sample.parquet",
    base_model_name: str = "microsoft/deberta-v3-small",
    output_dir: str = "outputs/models/treatment_deberta_lora",
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 2e-4,
    max_length: int = 256,
    random_seed: int = 42,
) -> dict:
    """Fine-tunes DeBERTa-v3 with LoRA specifically on multi-agent audited labels (treatment layer)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print("=== Multi-Agent SDP: Fine-Tuning DeBERTa-v3 on Audited Labels ===")
    print("=" * 65)
    print(f"Target Device: {device} | Base Model: {base_model_name}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 1. Load Data
    print(f"Loading dataset from: {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    label_map = {"bug": 1, "non-bug": 0}
    df["target"] = df["treatment_label"].map(label_map)

    # Stratified Split (70% train / 15% val / 15% test)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=random_seed,
        stratify=df["target"],
    )
    val_ratio = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=train_val_df["target"],
    )

    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = DefectTextDataset(
        texts=train_df["model_text"].tolist(),
        labels=train_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = DefectTextDataset(
        texts=val_df["model_text"].tolist(),
        labels=val_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    # 2. Setup DeBERTa-v3 with LoRA
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = max(int(total_steps * 0.1), 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    best_metrics = {}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loop.set_postfix(loss=total_loss / (step + 1))

        # Validation
        model.eval()
        val_targets, val_probas = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].numpy()

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probas = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                val_probas.extend(probas)
                val_targets.extend(targets)

        val_probas = np.array(val_probas)
        val_targets = np.array(val_targets)
        val_acc = accuracy_score(val_targets, (val_probas >= 0.5).astype(int))
        val_f1 = f1_score(val_targets, (val_probas >= 0.5).astype(int), average="macro")
        val_auc = roc_auc_score(val_targets, val_probas)

        print(f"--> Epoch {epoch+1} Results: Accuracy = {val_acc:.4f} | Macro F1 = {val_f1:.4f} | ROC-AUC = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {
                "epoch": epoch + 1,
                "val_accuracy": round(float(val_acc), 4),
                "val_macro_f1": round(float(val_f1), 4),
                "val_roc_auc": round(float(val_auc), 4),
                "model_dir": str(out_path),
            }
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)
            print(f"    [Checkpoint] Saved improved LoRA adapter to {out_path} (Val AUC: {best_val_auc:.4f})")

    # Save summary
    summary_path = Path("outputs/results/treatment/deberta_finetuning_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\nDeBERTa-v3 LoRA fine-tuning on audited labels completed! Best Val AUC: {best_val_auc:.4f}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa on Treatment Audited Labels")
    parser.add_argument("--dataset-path", default="data/processed/treatment_sample.parquet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    finetune_treatment_deberta(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
