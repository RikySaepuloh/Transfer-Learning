import json
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Mengatur perangkat untuk penggunaan GPU jika tersedia
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Membaca dataset dari file JSONL untuk generasi teks
class ChildMarriageDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, start=1):
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "messages" in entry:
                        messages = entry["messages"]
                        if len(messages) >= 2:
                            user_content = messages[0]["content"]
                            assistant_content = messages[1]["content"]
                            # Gabungkan percakapan user dan assistant untuk fine-tuning
                            self.data.append(f"User: {user_content} Assistant: {assistant_content}")
                        else:
                            print(f"Warning: Less than 2 messages in entry {entry}")
                    else:
                        print(f"Warning: Entry is not a valid dictionary with 'messages' key: {entry}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON at line {i}: {line}")
                    print(e)
                    continue  # Lewati baris yang menyebabkan kesalahan

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Memuat dataset
dataset = ChildMarriageDataset('child_marriage_data.jsonl')

# Menggunakan model GPT-2 dan tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Menambahkan pad_token menggunakan eos_token
tokenizer.pad_token = tokenizer.eos_token

# Memuat model GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Tokenisasi dan encoding untuk dataset
class ChildMarriageTensorDataset(Dataset):
    def __init__(self, input_texts):
        self.input_texts = input_texts

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        # Menggunakan padding dan truncation saat tokenisasi
        inputs = tokenizer(self.input_texts[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Membuat shifted labels untuk perhitungan loss
        labels = input_ids.clone()  # Salin input_ids menjadi labels
        labels[labels == tokenizer.pad_token_id] = -100  # Token padding tidak dihitung dalam loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels  # GPT-2 memerlukan 'labels' untuk menghitung loss
        }

# Siapkan data untuk fine-tuning
train_dataset = ChildMarriageTensorDataset(dataset.data)

# Mengatur argumen pelatihan
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Menggunakan batch kecil untuk model GPT-2
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Membuat Trainer untuk fine-tuning GPT-2
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Melatih model dari awal
trainer.train()

# Simpan model dan tokenizer setelah training
output_dir = './saved_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
