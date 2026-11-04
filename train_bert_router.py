"""Fine-tune DistilBERT (BERT family) to classify a request into one of 8 labels.
Runs on CPU in a few minutes.   Usage:  python train_bert_router.py
Output: ./bert_router/  (model + tokenizer + train_log.json)"""
import json
import random
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import make_train
from registry import LABELS

BASE, OUT, EPOCHS, LR, BS = "distilbert-base-uncased", "bert_router", 6, 5e-5, 16
random.seed(0)
torch.manual_seed(0)

rows = make_train(per_template=3)
l2i = {l: i for i, l in enumerate(LABELS)}
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=l2i)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

log, t0 = [], time.time()
for ep in range(EPOCHS):
    random.shuffle(rows)
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for i in range(0, len(rows), BS):
        batch = rows[i:i + BS]
        enc = tok([t for t, _ in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
        y = torch.tensor([l2i[l] for _, l in batch])
        out = model(**enc, labels=y)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        loss_sum += out.loss.item() * len(batch)
        correct += (out.logits.argmax(-1) == y).sum().item()
        total += len(batch)
    log.append({"epoch": ep + 1, "loss": round(loss_sum / total, 4), "train_acc": round(correct / total, 4)})
    print(log[-1])

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
json.dump({"base": BASE, "epochs": EPOCHS, "lr": LR, "batch_size": BS, "n_train": len(rows),
           "train_seconds": round(time.time() - t0, 1), "log": log}, open(f"{OUT}/train_log.json", "w"), indent=2)
print(f"saved to {OUT}/ in {time.time() - t0:.0f}s")
