from huggingface_hub import InferenceClient
from config import HF_API_KEY

models = [ 
"ByteDance/SDXL-Lightning",

"stabilityai/stable-diffusion-xl-base-1.0",

"stabilityai/sdxl-turbo",

"runwayml/stable-diffusion-v1-5",
]
client = InferenceClient(api_key=HF_API_KEY)

def generate_img(prompt):
    prompt = prompt.strip()
    if not prompt:
        print("Please enter a prompt to generate...")
    image = None
    
    for model in models:
        try:
            image = client.text_to_image(prompt=prompt, model=model)
            continue
        except Exception as e:
            print("Model not available. Using next from the list")
            print("Error: ", e)
            continue
        
    if image:
        file_name = "Generated_img.png"
        image.save(file_name)
        image.show()
    else:
        print("No model available for generation.")


while True:
    print("Enter a prompt to generate. \n")
    usr = input("> ")
    generate_img(usr)