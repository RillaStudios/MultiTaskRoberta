from datasets import load_dataset

# Load datasets
classification_dataset = load_dataset("imdb")  # Sentiment Classification
ner_dataset = load_dataset("conll2003")  # Named Entity Recognition
similarity_dataset = load_dataset("stsb_multi_mt", name="en")  # Sentence Similarity

print(classification_dataset, ner_dataset, similarity_dataset)