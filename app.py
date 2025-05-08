import json
import requests
from sentence_transformers import SentenceTransformer
import uuid

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import base64

from concurrent.futures import ThreadPoolExecutor, as_completed
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import io

from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/convert_vertor/", response_class=JSONResponse)
async def send_text(input: TextInput):
    
    image_path = 'a1.jpg'  # ระบุเส้นทางภาพที่ถูกต้อง
    image = cv2.imread(image_path)

    print(image.shape, "zeen")





if __name__ == "__main__":

    name_model_trans = "paraphrase-MiniLM-L6-v2"
    name_model_vector = "Salesforce/blip-image-captioning-base"
    name_model_detect = "yolov8n.pt"

    # สร้าง SentenceTransformer model
    model_sentence = SentenceTransformer(name_model_trans)
    # โหลดโมเดลและ processor
    processor = BlipProcessor.from_pretrained(name_model_vector)
    model_Blip = BlipForConditionalGeneration.from_pretrained(name_model_vector)
    # โหลดโมเดล YOLOv8
    model_detect = YOLO(name_model_detect)  

    list_model = [model_sentence, processor, model_Blip, model_detect]

    uvicorn.run(app, host="0.0.0.0", port=5325, log_level="info")
