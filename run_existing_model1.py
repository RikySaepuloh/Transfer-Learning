from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Mengatur perangkat untuk penggunaan GPU jika tersedia
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Memuat kembali model dan tokenizer dari disk
output_dir = './saved_model'
model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Fungsi untuk menghasilkan respons
def generate_response(user_input):
    # Tokenisasi input dengan padding
    inputs = tokenizer.encode(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Menghasilkan respons dari model yang sudah dilatih
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, 
                             no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, 
                             pad_token_id=tokenizer.pad_token_id)
    
    # Decode respons menjadi teks
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Contoh penggunaan model yang dimuat
user_question = "Kak, aku udah nggak bisa sekolah lagi karena nggak ada biaya. Daripada nganggur, mending nikah aja kan"
response = generate_response(user_question)
print(f"Predicted response: {response}")
