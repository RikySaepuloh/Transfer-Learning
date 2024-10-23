import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load tokenizer and pre-trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Load your dataset (assumes the JSON format like shown above)
dataset = load_dataset("json", data_files="child_marriage_data.json")

# Preprocess the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_dataset["train"]

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    for batch in train_loader:
        inputs = {k: v.to('cuda') for k, v in batch.items() if k in tokenizer.model_input_names}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
