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

def generate_response_streaming(user_input):
    # Menggunakan API chat.completions.create dengan max_tokens yang lebih besar
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that discusses child marriage prevention."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=500,  # Menyediakan batas token yang lebih besar
        temperature=0.5,  # Mengurangi temperature untuk respons yang lebih fokus
        top_p=0.95,
        n=1,
        stop=None,
        stream=True  # Mengaktifkan streaming
    )

    # Mengumpulkan potongan respons secara bertahap
    collected_response = []
    for chunk in response:
        if 'choices' in chunk:
            delta = chunk['choices'][0].get('delta', {})
            chunk_message = delta.get('content', '')
            if chunk_message:  # Validasi apakah chunk_message tidak kosong
                collected_response.append(chunk_message)
                print(chunk_message, end='', flush=True)  # Menampilkan hasil secara real-time

    # Menggabungkan semua potongan menjadi satu respons
    return ''.join(collected_response)

# Fungsi untuk melanjutkan respons jika terpotong (pagination)
def generate_full_response(user_input, max_tokens=500):
    collected_response = generate_response_streaming(user_input)
    
    # Loop untuk memeriksa apakah respons terpotong dan melanjutkan
    while True:
        # Cek apakah respons mendekati batas max_tokens, yang menandakan kemungkinan terpotong
        if len(collected_response.split()) < max_tokens - 10:
            break

        # Jika respons terpotong, minta model untuk melanjutkan
        last_few_words = " ".join(collected_response.split()[-10:])  # Ambil 10 kata terakhir
        user_input = f"Lanjutkan dari '{last_few_words}'"
        collected_response += generate_response_streaming(user_input)

    return collected_response

# Contoh penggunaan
user_question = "Kak, aku udah nggak bisa sekolah lagi karena biaya. Daripada nganggur, mending nikah aja kan?"
response = generate_full_response(user_question, max_tokens=500)
print(f"\nPredicted response: {response}")
