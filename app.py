from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn

import torch
from PIL import Image

import io
import base64
import uuid
from io import BytesIO
from starlette.responses import Response
from segmentation import get_yolov5, get_image_from_bytes

model = get_yolov5()

app = FastAPI()

@app.get("/")
def homepage():
    return {"MESSAGE": "Please go to /docs to test the app"}

@app.post('/predict')
async def detect_obj(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type='image/jpeg')

