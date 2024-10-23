from datasets import load_dataset
from transformers import AutoTokenizer, BertForQuestionAnswering, TrainingArguments, Trainer
import torch
import evaluate

# 1. Load dataset in JSON Lines format
dataset = load_dataset('json', data_files={'train': 'child_marriage_data.jsonl'})

# 2. Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 3. Preprocessing the dataset (Add start_positions and end_positions)
def preprocess_function(examples):
    questions = examples["question"]
    contexts = examples["context"]
    
    # Tokenize the questions and contexts
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map the answer start and end positions in the context
    start_positions = []
    end_positions = []

    for i in range(len(examples["answers"]["text"])):
        answer = examples["answers"]["text"][i]  # Ambil jawaban
        start_char = examples["answers"]["answer_start"][i]  # Ambil posisi mulai jawaban
        end_char = start_char + len(answer)

        offsets = inputs["offset_mapping"][i]
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end tokens
        start_token = 0
        end_token = 0

        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break
        
        start_positions.append(start_token)
        end_positions.append(end_token)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Load pre-trained BERT model for Question Answering
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 6. Compute metrics: Exact Match (EM) and F1 Score using evaluate library
metric = evaluate.load("squad")

def compute_metrics(pred):
    predictions = pred.predictions
    label_ids = pred.label_ids

    # Post-process the predictions and references
    formatted_predictions = [{"id": ex["id"], "prediction_text": pred} for ex, pred in zip(dataset["train"], predictions)]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset["train"]]

    # Calculate metrics (EM and F1)
    return metric.compute(predictions=formatted_predictions, references=references)

# 7. Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train the model
trainer.train()

# 9. Evaluate the model
eval_results = trainer.evaluate()

# 10. Print evaluation results
print(f"Evaluation results: {eval_results}")

# Optional: Test the model with a sample question
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1  # +1 to include the end token
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Example usage of answering a question
context = "Pernikahan anak adalah pernikahan yang terjadi di bawah usia 18 tahun."
question = "Apa itu pernikahan anak?"
answer = answer_question(question, context)
print(f"Question: {question}\nAnswer: {answer}")
