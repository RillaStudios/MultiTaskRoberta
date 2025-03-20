from data import load_data
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenization function
def tokenize_data(example, task):
    if task == "classification":
        return tokenizer(example["text"], padding="max_length", truncation=True)
    elif task == "token_classification":
        return tokenizer(example["tokens"], is_split_into_words=True, padding="max_length", truncation=True)
    elif task == "similarity":
        return tokenizer(example["sentence1"], example["sentence2"], padding="max_length", truncation=True)

# Apply tokenization
classification_dataset = load_data.classification_dataset.map(lambda x: tokenize_data(x, "classification"), batched=True)
ner_dataset = load_data.ner_dataset.map(lambda x: tokenize_data(x, "token_classification"), batched=True)
similarity_dataset = load_data.similarity_dataset.map(lambda x: tokenize_data(x, "similarity"), batched=True)
