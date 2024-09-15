import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Define your small dataset with context, question, and answer
data = [
    {"context": "The capital of France is Paris.", "question": "What is the capital of France?", "answer": "Paris."},
    {"context": "The book '1984' was written by George Orwell.", "question": "Who wrote '1984'?", "answer": "George Orwell."},
    {"context": "The largest planet in our solar system is Jupiter.", "question": "What is the largest planet in our solar system?", "answer": "Jupiter."},
    {"context": "There are seven continents on Earth.", "question": "How many continents are there?", "answer": "Seven."},
    {"context": "The chemical symbol for water is H2O.", "question": "What is the chemical symbol for water?", "answer": "H2O."},
    {"context": "I'm a model named terminator, who the hell are you?" , "question": "who are you", "answer": "I'm a model named terminator, who the hell are you"}
]

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(data)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Tokenize the dataset
def tokenize_function(examples):
    inputs = [f"context: {c} question: {q}" for c, q in zip(examples['context'], examples['question'])]
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to='none'  # Disable W&B logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # For simplicity, use the same dataset for evaluation
)

# Train the model
trainer.train()

# Simulated document index
documents = [
    "The capital of France is Paris.",
    "The book '1984' was written by George Orwell.",
    "The largest planet in our solar system is Jupiter.",
    "There are seven continents on Earth.",
    "The chemical symbol for water is H2O.",
    "Waqar is a simple person who tried to do his best and loves the color red.",
    "I'm a model named terminator, who the hell are you?"
]

# Improved retrieval based on keyword matching
def retrieve_relevant_documents(question):
    # For better results, use more sophisticated methods for document retrieval in real-world scenarios
    relevant_docs = [doc for doc in documents if any(keyword.lower() in doc.lower() for keyword in question.lower().split())]
    return relevant_docs

# Define the question-answering function
def answer_question(question, max_len=100):
    relevant_docs = retrieve_relevant_documents(question)
    if not relevant_docs:
        return "No relevant documents found."

    context = ' '.join(relevant_docs)  # Combine retrieved documents

    input_text = f"context: {context} question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    output_ids = model.generate(input_ids,
                                num_beams=8,
                                no_repeat_ngram_size=2,
                                min_length=10,
                                max_length=max_len,
                                early_stopping=True)

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return answer

# Example usage
question = "What is the largest planet in the solar system?"
print("Question:", question)
print("Answer:", answer_question(question))

