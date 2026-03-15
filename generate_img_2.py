from huggingface_hub import InferenceClient
from config import HF_API_KEY
from PIL import Image, ImageEnhance, ImageFilter

models = [ 
"ByteDance/SDXL-Lightning",

"stabilityai/stable-diffusion-xl-base-1.0",

"stabilityai/sdxl-turbo",

"runwayml/stable-diffusion-v1-5",
]
client = InferenceClient(api_key=HF_API_KEY)

def enhance_img(img):
    # augments the brightness
    new_img = img.copy()
    brightness = ImageEnhance.Brightness(new_img)
    new_img = brightness.enhance(1.2)
    contrast = ImageEnhance.Contrast(new_img)
    new_img = contrast.enhance(1.3)
    new_img = new_img.filter(ImageFilter.GaussianBlur(radius=5))
    
    return new_img
    
    

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
        return image
    else:
        print("No model available for generation.")


while True:
    print("Enter a prompt to generate. \n")
    file_name1 = "Generated_img.png"
    file_name2 = "Generated_img_en.png"
    usr = input("> ")
    img = generate_img(usr)
    img.save(file_name1)
    
    img_en = enhance_img(img)
    img_en.save(file_name2)
    
    img_en.show()