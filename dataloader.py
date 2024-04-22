# =============================================================================================
# 1. Install Required Libraries
# First, make sure you have the necessary libraries installed: pip install transformers torch
# =============================================================================================

# =============================================================================================
# 2. Prepare the Dataset
# Assuming you have a dataset in a simple format where each line is a text followed by its label 
# (1 for 'financial reporting', 0 for 'not financial reporting'), you can create a custom dataset class.
# =============================================================================================
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class FinanceDataset(Dataset):
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
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================================
# 3. Load the Tokenizer and Initialize the Dataset
# =============================================================================================

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Example data
texts = ["This is about financial reporting.", "This is not related to finance.", ...]
labels = [1, 0, ...]

# Create the dataset
dataset = FinanceDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# =============================================================================================
# 4. Load the Model and Modify for Classification
# You'll need to modify the MPNet model for classification by adding a classification head. 
# You can do this by using AutoModelForSequenceClassification.
# =============================================================================================
from transformers import AutoModelForSequenceClassification, AdamW

model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-mpnet-base-v2", num_labels=2)

# =============================================================================================
# 5. Training Loop: Set up the training loop using PyTorch.
# =============================================================================================
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3  # Define the number of epochs

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# =============================================================================================
# 6. Evaluation
# To evaluate the model, ensure you separate some data as a test set and measure the accuracy of the model on this unseen data.
# =============================================================================================

# Switch model to evaluation mode
model.eval()
# Add code to calculate accuracy on the test set
