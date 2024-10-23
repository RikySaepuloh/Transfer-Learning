from transformers import BertTokenizer, BertForQuestionAnswering
from datasets import load_dataset
import torch
import evaluate

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files="child_marriage_data.jsonl")
metric = evaluate.load("squad")

# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['question'], examples['context'], max_length=512, truncation=True, padding="max_length")
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Evaluation function
def evaluate_model():
    model.eval()
    predictions = []
    references = []

    for example in tokenized_dataset['train']:
        inputs = {
            "input_ids": torch.tensor([example['input_ids']]),
            "attention_mask": torch.tensor([example['attention_mask']])
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            all_tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            
            # Get the most probable start and end of the answer
            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits)
            answer = tokenizer.convert_tokens_to_string(all_tokens[answer_start:answer_end + 1])
        
        # Store predictions and references in expected format
        predictions.append({
            "id": example["id"],
            "prediction_text": answer
        })
        
        references.append({
            "id": example["id"],
            "answers": {
                "text": example["answers"]["text"],
                "answer_start": example["answers"]["answer_start"]
            }
        })
    
    # Compute the metrics
    results = metric.compute(predictions=predictions, references=references)
    return results

# Call evaluate function and print the results
result = evaluate_model()
print(result)

# Hardcoded questions and context
hardcoded_context = "Pernikahan anak adalah pernikahan yang terjadi di bawah usia 18 tahun."
hardcoded_questions = [
    "Apa itu pernikahan anak?",
    "Apa penyebab pernikahan anak?"
]

# Function to predict answers for hardcoded questions
def answer_hardcoded_questions(context, questions):
    for question in questions:
        inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits)
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end + 1]))

            print(f"Question: {question}\nAnswer: {answer}\n")

# Call the function to get answers for hardcoded questions
answer_hardcoded_questions(hardcoded_context, hardcoded_questions)
