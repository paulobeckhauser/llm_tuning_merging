from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm  # For progress bar
import os
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['HF_HOME'] = '/dtu/blackhole/00/214080/'

# Replace "your_huggingface_token" with your actual token
login("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")

# Initialize the model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float16,
        device_map="auto"
    )

# # Load the model checkpoint and dispatch it to the appropriate device
# model = load_checkpoint_and_dispatch(
#     model, 
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     device_map="auto"
# )

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3"
)


# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

dataset = load_dataset('cais/mmlu', 'all')

def evaluate_model(model, tokenizer, dataset, max_new_tokens=8):
    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(dataset, desc="Evaluating", unit="sample"):
        question = example["question"]
        choices = example["choices"]
        label = example["answer"]  # Ensure this is the correct index

        prompt = f"Question:\n{question}\nChoices:\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}\nAnswer just the letter (A, B, C, D), don't explain"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, pad_token_id=tokenizer.eos_token_id)  # Explicitly set pad_token_id)
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
large_dataset = dataset["test"]
accuracy = evaluate_model(model, tokenizer, large_dataset, max_new_tokens=8)
print(f"Accuracy on subset: {accuracy:.2%}")