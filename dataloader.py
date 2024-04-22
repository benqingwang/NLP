import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize the text
        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze()  # Remove batch dimension
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example usage
texts = ["Hello, world!", "Machine learning is fun."]
labels = [0, 1]  # Example binary labels for each text
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
