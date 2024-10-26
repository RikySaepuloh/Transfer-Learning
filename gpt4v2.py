import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Set API key dari OpenAI melalui dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Membaca dataset dari file JSONL untuk digunakan dalam percakapan
class ChildMarriageDataset:
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
                            self.data.append((user_content, assistant_content))
                        else:
                            print(f"Warning: Less than 2 messages in entry {entry}")
                    else:
                        print(f"Warning: Entry is not a valid dictionary with 'messages' key: {entry}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON at line {i}: {line}")
                    print(e)
                    continue  # Lewati baris yang menyebabkan kesalahan

# Memuat dataset
dataset = ChildMarriageDataset('child_marriage_data.jsonl')

# Fungsi untuk menghasilkan respons dari GPT-4 menggunakan OpenAI API
def generate_response(user_input):
    # Menggunakan API chat.completions.create
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that discusses child marriage prevention."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=300,  # Meningkatkan batas token agar respons lebih panjang
        temperature=0.5,
        top_p=0.95,
        n=1,
        stop=None  # Kamu bisa tambahkan pola tertentu di sini jika ingin respons berhenti pada kondisi spesifik
    )

    # Memeriksa apakah respons valid dan mengakses kontennya
    choices = response.choices
    if choices and hasattr(choices[0], 'message'):
        return choices[0].message.content  # Mengakses atribut `content` dari objek `message`
    else:
        raise ValueError("Unexpected response format from OpenAI API")

# Menggunakan dataset untuk menghasilkan contoh interaksi dengan model
# for i, (user_question, expected_answer) in enumerate(dataset.data[:5], start=1):  # Mengambil 5 contoh pertama dari dataset
#     print(f"Example {i}:")
#     print(f"User: {user_question}")
#     response = generate_response(user_question)
#     print(f"Assistant: {response}")
#     print(f"Expected (from dataset): {expected_answer}")
#     print("-" * 50)

# Contoh penggunaan langsung (bukan dari dataset)
user_question = "Kak, keluarga aku nggak punya uang, terus ada orang yang mau menikah sama aku dan bisa bantuin ekonomi keluarga. Kalau aku nikah muda, keluargaku bisa jadi lebih baik, kan?"
response = generate_response(user_question)
print(f"Predicted response: {response}")
