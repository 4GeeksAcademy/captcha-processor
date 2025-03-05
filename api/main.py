from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import random
import string

# API initialization and model load
app = FastAPI()
# model = torch.hub.load("ultralytics/yolov5", "custom", path="4geeks_model.pt", force_reload=True)  

def mock_prediction():
    """Generate a random 5-6 character CAPTCHA string as a mock prediction."""
    length = random.randint(5, 6)  # Random CAPTCHA length
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def read_image(file: UploadFile):
    """ Convert uploaded file to OpenCV image """
    image_stream = BytesIO(file.file.read())
    image = Image.open(image_stream).convert("RGB")
    return np.array(image)

'''Usaremos esto cuando importemos el modelo'''
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
   
#     image = read_image(file)

#     # Convert image to proper format for YOLO
#     results = model(image)

#     # Extract predictions
#     predictions = results.pandas().xyxy[0]  
#     extracted_text = "".join(predictions['name'].tolist()) 

#     return {"captcha_text": extracted_text}


'''Esto es el mockup'''
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Simulate processing the image and returning a dummy prediction
    captcha_text = mock_prediction()
    return {"captcha_text": captcha_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)