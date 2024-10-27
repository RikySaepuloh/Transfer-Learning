from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Mengatur perangkat untuk penggunaan GPU jika tersedia
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Memuat kembali model dan tokenizer dari disk
output_dir = './saved_model'
model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Tambahkan pad_token jika belum ada
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# Fungsi untuk menghasilkan respons
def generate_response(user_input):
    # Format percakapan yang lebih jelas untuk GPT-2
    input_prompt = f"User: {user_input}\nAssistant: "
    
    # Tokenisasi input dengan padding dan attention mask
    inputs = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Menghasilkan respons dari model yang sudah dilatih
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode respons menjadi teks
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Menghapus pertanyaan dari jawaban
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    return response

# Contoh penggunaan model yang dimuat
user_question = "Kak, keluarga aku nggak punya uang, terus ada orang yang mau menikah sama aku dan bisa bantuin ekonomi keluarga. Kalau aku nikah muda, keluargaku bisa jadi lebih baik, kan?"
response = generate_response(user_question)
print(f"Predicted response: {response}")
