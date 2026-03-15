from PIL import ImageDraw, Image
import io
from config import HF_API_KEY
import requests

MODEL="facebook/detr-resnet-50"
API_URL=f"https://router.huggingface.co/hf-inference/models/{MODEL}"
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "image/jpeg"
}


def detect_img(img_path):
    img_bytes = None
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    req = requests.post(API_URL, headers=headers, data=img_bytes)

    if req.status_code != 200:
        print("An error occured.")
        return None
    
    return req.json()

def draw_box(detection_data, img_path):
    f = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(f)

    # draws the rectangles
    for obj in detection_data:
        score = obj['score']
        
        if score < 0.5:
            continue
        box = obj['box']
        label = obj['label']
        x1 = box['xmin']
        y1 = box['ymin']
        x2 = box['xmax']
        y2 = box['ymax']
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        draw.text((x1,y1), label, fill="red")
    f.save("output.png")

file_p = "./assets/dogs.jpeg"

data = detect_img(file_p)

draw_box(data, file_p)