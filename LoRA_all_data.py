# main.py

# Import necessary libraries
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import re
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Step 1: Login to Hugging Face
# Replace "your_huggingface_token" with your actual token
login("")

# Step 2: Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # You can also try torch.bfloat16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Step 3: Load the model with torch_dtype=torch.float16
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    #torch_dtype=torch.float16,  # Ensure model parameters are in float16
)

# Step 4: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    trust_remote_code=True,
)

# Set the pad token to an existing token to avoid resizing embeddings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 5: Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)


# Step 6: Enable gradient checkpointing and set use_cache to False
model.config.use_cache = False
model.gradient_checkpointing_enable()


# Step 7: Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


# Step 9: Create the custom dataset
class AlgebraDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        question = data["question"]
        choices = data["choices"]

        # Convert integer answer to letter
        answer_idx = data["answer"]
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(choices):
            raise ValueError(f"Invalid answer index: {answer_idx}")
        label = chr(65 + answer_idx)  # Convert 0->A, 1->B, 2->C, 3->D

        # Create the prompt
        prompt = (
            f"Question:\n{question}\n"
            f"Choices:\n"
            f"A: {choices[0]}\n"
            f"B: {choices[1]}\n"
            f"C: {choices[2]}\n"
            f"D: {choices[3]}\n"
            "Answer just the letter (A, B, C, D), don't explain.\n"
        )

        # The target answer is the label (e.g., "B")
        answer = label

        # Concatenate prompt and answer
        full_text = prompt + answer

        # Tokenize the full text
        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )

        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        # Create labels: -100 for prompt tokens, input_ids for answer tokens
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer(prompt, truncation=True, max_length=512)["input_ids"])
        labels[:prompt_len] = -100  # Ignore prompt tokens in the loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": question,
            "choices": choices,
            "answer": label  # Now a letter "A", "B", "C", or "D"
        }

# Step 10: Load datasets
dataset = load_dataset('cais/mmlu', 'all')
train_dataset = dataset['test']
test_dataset = dataset['validation']

# Create dataset instances
train_data = AlgebraDataset(train_dataset, tokenizer)
test_data = AlgebraDataset(test_dataset, tokenizer)

# Step 11: Define training arguments
training_args = TrainingArguments(
    output_dir="./mistral-lora-law",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4, #No need for accumulation since we use 32 as batch size
    evaluation_strategy="no",  # Disable evaluation during training
    save_strategy="no",        # Disable saving checkpoints
    logging_steps=50,
    num_train_epochs=5,        # Adjust as needed
    learning_rate=5e-4,
    fp16=True,                 # Enable automatic mixed precision
    torch_compile=False,
    push_to_hub=False,
    gradient_checkpointing=True,  # Ensure gradient checkpointing is disabled
)

# Step 12: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Train the model
train_result = trainer.train()

# Log training loss and other metrics
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()


# Step 13: Train the model
trainer.train()

# Save the model
model.save_pretrained("./mistral-finetuned-all")

# Save the tokenizer
tokenizer.save_pretrained("./mistral-finetuned-all")


# Step 14: Evaluate the model
def evaluate_model(model, tokenizer, dataset, max_new_tokens=8):
    correct_predictions = 0
    total_predictions = 0
    results = []

    for example in tqdm(dataset, desc="Evaluating", unit="sample"):
        question = example["question"]
        choices = example["choices"]
        correct_label = example["answer"]

        prompt = (
            f"Question:\n{question}\n"
            f"Choices:\n"
            f"A: {choices[0]}\n"
            f"B: {choices[1]}\n"
            f"C: {choices[2]}\n"
            f"D: {choices[3]}\n"
            "Answer just the letter (A, B, C, D), don't explain.\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text[len(prompt):].strip()

        # Extract the first occurrence of a letter A-D
        match = re.search(r'[A-Da-d]', response_text)
        if match:
            predicted_label = match.group(0).upper()

            is_correct = predicted_label == correct_label
            correct_predictions += is_correct

            # Store results for logging
            results.append({
                "Question": question,
                "Choices": choices,
                "Correct Answer": correct_label,
                "Predicted Answer": predicted_label,
                "Correct?": "✔️" if is_correct else "❌"
            })

        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Display results as a DataFrame
    df_results = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df_results)

    print(f"\nFinal Accuracy on Test Dataset: {accuracy:.2%}")
    return accuracy

# Step 15: Evaluate the model on the test dataset
accuracy = evaluate_model(model, tokenizer, test_data, max_new_tokens=8)

