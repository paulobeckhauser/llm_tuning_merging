
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

def main():
    
    # set the model as tiny llama
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # get the tokenizer from the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create a pipeline for text generation
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0) # Set device to 0 for GPU

    # Generate text based on a prompt
    prompt = "Who is Ada Lovelace?"
    generated_text = generator(prompt, max_length=50)

    # print the result
    print(generated_text[0]['generated_text'])





if __name__ == "__main__":
    main()
