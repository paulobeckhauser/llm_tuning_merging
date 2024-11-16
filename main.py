# from dotenv import load_dotenv
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login

def main():

    # # Check PyTorch version
    # print("PyTorch version:", torch.__version__)

    # # Check for GPU
    # if torch.cuda.is_available():
    #     print("GPU available:", torch.cuda.get_device_name(0))
    # else:
    #     print("No GPU detected.")

    # # Load environment variables from .env file
    # load_dotenv()

    # # Access the Hugging Face API key
    # hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    # if not hf_api_key:
    #     raise ValueError("Hugging Face API key not found. Make sure it's set in the .env file.")

    # print("Hugging Face API Key loaded successfully.")

    # # Login to Hugging Face
    # try:
    #     login(hf_api_key)
    #     print("Logged into Hugging Face successfully.")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to log in to Hugging Face: {e}")

    # # Load the tokenizer and model
    # model_name = "mistralai/Mistral-7B-v0.1"
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    #     print("Tokenizer loaded successfully.")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to load tokenizer: {e}")

    # try:
    #     model = AutoModelForCausalLM.from_pretrained(model_name, token=True, device_map="auto")
    #     print("Model loaded successfully.")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to load model: {e}")

    # # Ensure the model is assigned to the device (optional if device_map="auto" works)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    #     model = model.to(device)
    #     print(f"Model moved to device: {device}")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to move model to device: {e}")

    # set the model as tiny llama
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_name = "mistralai/Mistral-7B-v0.1"

    # # get the tokenizer from the model
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Load the model
    # model = AutoModelForCausalLM.from_pretrained(model_name)

    # # Create a pipeline for text generation
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0) # Set device to 0 for GPU

    # # Generate text based on a prompt
    # prompt = "Who is Ada Lovelace?"
    # generated_text = generator(prompt, max_length=50)

    # # print the result
    # print(generated_text[0]['generated_text'])
    # Load model directly


    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")