import torch
import torch.nn as nn
from transformers import RobertaModel

class MultiTaskRoBERTa(nn.Module):
    def __init__(self, model_name="roberta-base", num_resume_classes=5, num_job_classes=5, num_resume_labels=9,
                 num_job_labels=9):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)

        # Separate classification heads for resume and job posting sections
        self.resume_classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_resume_classes)
        )

        self.job_classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_job_classes)
        )

        # Separate token classification (NER) heads
        self.resume_ner_head = nn.Linear(768, num_resume_labels)
        self.job_ner_head = nn.Linear(768, num_job_labels)

        # Shared sentence similarity head
        self.similarity_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, task_type="resume_classification"):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        if task_type == "resume_classification":
            return self.resume_classification_head(pooled_output)
        elif task_type == "job_classification":
            return self.job_classification_head(pooled_output)
        elif task_type == "resume_ner":
            return self.resume_ner_head(outputs.last_hidden_state)  # Per-token output
        elif task_type == "job_ner":
            return self.job_ner_head(outputs.last_hidden_state)  # Per-token output
        elif task_type == "similarity":
            return self.similarity_head(pooled_output)  # Regression output

