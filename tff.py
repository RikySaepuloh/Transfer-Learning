# Mengimpor library yang diperlukan
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Mengatur perangkat untuk penggunaan GPU jika tersedia
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Membaca dataset dari file JSONL
class ChildMarriageDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    entry = json.loads(line.strip())
                    messages = entry["messages"]
                    user_content = messages[0]["content"]
                    assistant_content = messages[1]["content"]
                    self.data.append((user_content, assistant_content))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(e)
                    continue  # Lewati baris yang menyebabkan kesalahan

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Memuat dataset
dataset = ChildMarriageDataset('dataset.jsonl')

# Membagi dataset menjadi data pelatihan dan data pengujian
train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

# Menggunakan tokenizer dari model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenisasi dan encoding data
def encode_data(data):
    inputs = tokenizer([entry[0] for entry in data], padding=True, truncation=True, return_tensors="pt")
    outputs = tokenizer([entry[1] for entry in data], padding=True, truncation=True, return_tensors="pt")
    return inputs, outputs['input_ids'], outputs['attention_mask']

train_inputs, train_outputs, train_attention_masks = encode_data(train_data)
test_inputs, test_outputs, test_attention_masks = encode_data(test_data)

# Membuat DataLoader
train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_outputs)
test_dataset = torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_outputs)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Memuat model BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Mengatur argumen pelatihan
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Membuat Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Melatih model
trainer.train()

# Menggunakan model untuk memberikan respons
def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Menjalankan model untuk mendapatkan keluaran
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Mengambil respons berdasarkan output model
    predicted_index = torch.argmax(outputs.logits, dim=-1).item()
    
    # Kembalikan respons sesuai prediksi
    response_index = predicted_index  # Misalnya, kita ambil prediksi dan mapping ke respons
    return train_outputs[response_index]

# Pertanyaan untuk diuji
user_question = "Kak, keluarga aku nggak punya uang, terus ada orang yang mau menikah sama aku dan bisa bantuin ekonomi keluarga. Kalau aku nikah muda, keluargaku bisa jadi lebih baik, kan?"
response = generate_response(user_question)
print(response)
