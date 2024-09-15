import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import textwrap

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-large').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-large')

# Define the question-answering function
def ask_question(context, question, max_len):
    # Preprocess the question by including context and using the right format for question answering
    t5_prepared_question = f"question: {question.strip()} context: {context.strip()}"
    wrapped_t5_prepared_question = textwrap.fill(t5_prepared_question, width=70)
    print(wrapped_t5_prepared_question)
    
    # Tokenize the input question and context
    tokenized_question = tokenizer.encode(t5_prepared_question, return_tensors="pt").to(device)
    
    # Generate an answer using the model
    answer_ids = model.generate(tokenized_question,
                                num_beams=8,
                                no_repeat_ngram_size=3,
                                min_length=20,
                                max_length=max_len,
                                early_stopping=True)
    
    # Decode the generated answer
    output = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return output

# Example usage: Providing the context along with the question
context = """
We hold these truths to be self-evident, that all men are created equal, 
that they are endowed by their Creator with certain unalienable Rights, 
that among these are Life, Liberty, and the pursuit of Happiness.
"""

question = "What are the unalienable rights mentioned in the Declaration of Independence?"

print("Question:", question.strip())
answer = ask_question(context, question, 50)
wrapped_answer = textwrap.fill(answer, width=70)
print("\nAnswer:\n", wrapped_answer)

