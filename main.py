from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datasets import load_dataset
from tqdm import tqdm  # For progress bar
from dotenv import load_dotenv
import os
from evaluate import load
from torch.utils.data import DataLoader

# Load environment variables from .env file
load_dotenv()

# Set the custom cache directory
os.environ['HF_HOME'] = os.getenv('HF_HOME')

# Get the Hugging Face API key from environment variables
token = os.getenv("HUGGING_FACE_API_KEY")

# Ensure CUDA is available
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

# Replace "your_huggingface_token" with your actual token
# login("token")
login("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")


# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically distributes the model across available devices
    token="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"
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    token="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"
)

# Set the pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the accuracy metric
accuracy = load("accuracy")

dataset = load_dataset('cais/mmlu', 'abstract_algebra')

def evaluate_model(model, tokenizer, dataset, max_new_tokens=50):
    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(dataset, desc="Evaluating", unit="sample"):
        question = example["question"]
        choices = example["choices"]
        label = example["answer"]  # Ensure this is the correct index

        prompt = f"Question:\n{question}\nChoices:\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}\nAnswer just the letter (A, B, C, D), don't explain"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text[len(prompt) + 1:]
        response_text = response_text.strip()[0]
        print(response_text)

        # Map response to choice
        if "A" in response_text and label == 0:
            correct_predictions += 1
        elif "B" in response_text and label == 1:
            correct_predictions += 1
        elif "C" in response_text and label == 2:
            correct_predictions += 1
        elif "D" in response_text and label == 3:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# Subset of test data for evaluation
large_dataset = dataset["test"].select(range(20))
accuracy = evaluate_model(model, tokenizer, large_dataset, max_new_tokens=100)
print(f"Accuracy on subset: {accuracy:.2%}")