from flask import Flask, request, jsonify
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Define paths
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
model_path = "./trained_model"

# Initialize the tokenizer
Ltokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the saved LoRA config
lora_config = LoraConfig.from_pretrained(model_path)

# Load the fine-tuned model with LoRA configuration
model = get_peft_model(base_model, lora_config)

text_gen = pipeline(
    task="text-generation",
    model=base_model,
    tokenizer=Ltokenizer,
    max_length=512,
    device=0 if torch.cuda.is_available() else -1,
)

# Define the introduction message outside the loop
introduction = "Your name is Virtual Psych. You are a chatbot built for mental health assistance. Answer the question, it should be concise and encouraging."

# Initialize conversation history as a list with a maximum length
max_history_length = 5
conversation_history = []

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']

    # Add user input to the conversation history
    conversation_history.append(f"You: {user_input}")

    # Trim the conversation history to the maximum length
    if len(conversation_history) > max_history_length:
        conversation_history.pop(0)  # Remove the oldest message

    # Generate a response
    context = "\n".join(conversation_history)
    prompt = f"{introduction}\nQuestion: {user_input}\nContext:\n{context}\n"
    response = text_gen(prompt)[0]["generated_text"]
    pattern = r"(Virtual Psych:|Answer:)"  # Define the pattern to match either "Virtual Psych:" or "Answer:"
    split_text = re.split(pattern, response, 1)

    if len(split_text) > 2:
        response = split_text[2].strip()
        conversation_history.append(f"Virtual Psych: {response}")
        return jsonify({'response': response})
    else:
        return jsonify({'response': "Bot response is incomplete or missing."})

if __name__ == '__main__':
    app.run(debug=True)