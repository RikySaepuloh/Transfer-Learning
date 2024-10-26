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

def generate_response(user_input, max_tokens=300):
    # Menggunakan API chat.completions.create dengan max_tokens yang lebih besar
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that discusses child marriage prevention."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=max_tokens,
        temperature=0.5,  # Mengurangi temperature untuk respons yang lebih fokus dan tidak berbelit
        top_p=0.95,
        n=1,
        stop=None
    )

    # Memeriksa apakah respons valid dan mengakses kontennya
    choices = response.choices
    if choices and hasattr(choices[0], 'message'):
        return choices[0].message.content
    else:
        raise ValueError("Unexpected response format from OpenAI API")

# Fungsi untuk melanjutkan respons jika terpotong
def continue_response(previous_response, last_few_words, max_tokens=300):
    user_input_followup = f"Lanjutkan dari '{last_few_words}'"
    return generate_response(user_input_followup, max_tokens=max_tokens)

# Contoh penggunaan
user_question = "Kak, aku udah nggak bisa sekolah lagi karena biaya. Daripada nganggur, mending nikah aja kan?"
response = generate_response(user_question, max_tokens=300)

print(f"Predicted response: {response}")