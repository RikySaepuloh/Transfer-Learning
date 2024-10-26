import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

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
                            self.data.append((user_content, assistant_content))
                        else:
                            print(f"Warning: Less than 2 messages in entry {entry}")
                    else:
                        print(f"Warning: Entry is not a valid dictionary with 'messages' key: {entry}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON at line {i}: {line}")
                    print(e)
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Memuat dataset
dataset = ChildMarriageDataset('child_marriage_data.jsonl')

# Menggunakan IndoBERT dan tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')

# Tokenisasi dan encoding untuk dataset
class ChildMarriageTensorDataset(Dataset):
    def __init__(self, input_texts):
        self.input_texts = input_texts

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        # Tokenisasi input dan output
        inputs = tokenizer(self.input_texts[idx][0], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        outputs = tokenizer(self.input_texts[idx][1], return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = outputs['input_ids'].squeeze()

        labels[labels == tokenizer.pad_token_id] = -100  # Token padding tidak dihitung dalam loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Siapkan data untuk fine-tuning
train_dataset = ChildMarriageTensorDataset(dataset.data)

# Memuat model IndoBERT untuk klasifikasi atau pemahaman teks
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=2).to(device)

# Mengatur argumen pelatihan
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Membuat Trainer untuk fine-tuning IndoBERT
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Melatih model
trainer.train()

# Simpan model dan tokenizer setelah training
output_dir = './saved_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Memuat kembali model dan tokenizer dari disk
model = BertForSequenceClassification.from_pretrained(output_dir).to(device)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Fungsi untuk menghasilkan respons
def generate_response(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Menghasilkan respons dari model yang sudah dilatih
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, 
                             no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, 
                             pad_token_id=tokenizer.pad_token_id)
    
    # Decode respons menjadi teks
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Contoh penggunaan model yang dimuat
user_question = "Kak, keluarga aku nggak punya uang, terus ada orang yang mau menikah sama aku dan bisa bantuin ekonomi keluarga. Kalau aku nikah muda, keluargaku bisa jadi lebih baik, kan?"
response = generate_response(user_question)
print(f"Predicted response: {response}")
