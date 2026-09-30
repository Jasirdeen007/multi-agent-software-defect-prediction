from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, roc_auc_score

from src.baseline.data_splitter import load_dataset, split_dataset


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


def finetune_deberta(
    model_name: str = "microsoft/deberta-v3-small",
    dataset_path: str = "bughubs_canonical.parquet",
    sample_size: int = 50000,
    output_dir: str = "outputs/models/deberta_lora",
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    lr: float = 2e-4,
    max_length: int = 256,
    random_seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Fine-Tuning DeBERTa-v3 with LoRA ===")
    print(f"Target Device: {device} | Base Model: {model_name}")

    # 1. Load Data
    df = load_dataset(dataset_path, sample_size=sample_size, random_seed=random_seed)
    train_df, val_df, _ = split_dataset(df, random_seed=random_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    # 2. Setup Model with LoRA
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Training Loop
    best_val_auc = 0.0
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

        # 5. Validation Evaluation
        model.eval()
        val_preds, val_targets, val_probas = [], [], []
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
        val_auc = roc_auc_score(val_targets, val_probas)

        print(f"Epoch {epoch+1} Validation Results: Accuracy = {val_acc:.4f} | ROC-AUC = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)
            print(f"--> Saved improved model checkpoint to {out_path} (Best Val AUC: {best_val_auc:.4f})")

    print("\nFine-tuning completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa with LoRA")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=50000)
    args = parser.parse_args()

    finetune_deberta(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        sample_size=args.sample_size,
    )
