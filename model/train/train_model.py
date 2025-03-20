import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

from model.resume_roberta import MultiTaskRoBERTa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskRoBERTa().to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

classification_loss_fn = nn.CrossEntropyLoss()
token_classification_loss_fn = nn.CrossEntropyLoss()
similarity_loss_fn = nn.MSELoss()

scaler = GradScaler()

def train_step(batch, task_type):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()

    with autocast():  # Mixed precision
        outputs = model(input_ids, attention_mask=attention_mask, task_type=task_type)
        if task_type == "classification":
            loss = classification_loss_fn(outputs, labels)
        elif task_type == "token_classification":
            loss = token_classification_loss_fn(outputs.view(-1, 9), labels.view(-1))
        elif task_type == "similarity":
            loss = similarity_loss_fn(outputs.squeeze(), labels.float())

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
