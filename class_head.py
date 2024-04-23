# ==================================================================================
# Step 1: Installation
# ==================================================================================
# First, ensure you have the necessary libraries installed:
# pip install transformers torch

# ==================================================================================
# Step 2: Prepare the Dataset
# ==================================================================================
import torch
from torch.utils.data import Dataset, DataLoader

class RiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove the batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example data
texts = ["Company faced a severe backlash from a data breach.",
         "New office opened in Berlin."]
labels = [1, 0]  # 1 = related, 0 = not related

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')

# Create dataset
dataset = RiskDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# ==================================================================================
# Step 3: Define the Model and Training Loop
# ==================================================================================
# Set up the model for sequence classification:

from transformers import AutoModelForSequenceClassification, AdamW
model = AutoModelForSequenceClassification.from_pretrained('microsoft/mpnet-base', num_labels=2)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
num_epochs = 3
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# ==================================================================================
# Step 4: Save the Model and Tokenizer
# ==================================================================================
model.save_pretrained('/path/to/save/model/')
tokenizer.save_pretrained('/path/to/save/tokenizer/')

# ==================================================================================
# Step 5: Load and Use the Model to Make Predictions
# ==================================================================================
# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('/path/to/save/model/')
tokenizer = AutoTokenizer.from_pretrained('/path/to/save/tokenizer/')

# Prepare the model for evaluation (set to evaluation mode)
model.eval()

# Example text for classification
text_to_classify = "The firm is under scrutiny after the financial report."

# Tokenize the text
encoded_input = tokenizer(text_to_classify, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Predict
with torch.no_grad():
    outputs = model(**encoded_input)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Print the predictions
print("Probabilities:", predictions)
