import torch

def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    encodings["labels"] = torch.tensor(labels, dtype=torch.float).unsqueeze(1)  # Regression labels
    return encodings
