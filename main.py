import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

def main():

    load_dotenv()
    HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
    print(HUGGING_FACE_API_KEY)



if __name__ == "__main__":
    main()