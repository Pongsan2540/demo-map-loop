'''
import json
import requests
from sentence_transformers import SentenceTransformer
import uuid

# สร้าง SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ข้อความที่ต้องการแปลงเป็นเวกเตอร์
text = "two-door sedan"

# แปลงข้อความเป็นเวกเตอร์
vector = model.encode(text)

# สร้างข้อมูลที่ต้องการส่งไปยัง Elasticsearch
data = {
    "vector": vector.tolist(),  # เปลี่ยนเวกเตอร์เป็น list เพื่อให้ส่งได้
    "text": text
}

# สร้าง ID ใหม่ด้วย uuid เพื่อหลีกเลี่ยงการทับข้อมูลเดิม
document_id = str(uuid.uuid4())  # ใช้ UUID สำหรับ ID ที่ไม่ซ้ำ

# ส่งข้อมูลไปยัง Elasticsearch
url = f"http://localhost:9200/my-index/_doc/{document_id}"
headers = {'Content-Type': 'application/json'}
auth = ('elastic', 'changeme')

# ส่ง POST request
response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))

# แสดงผลการตอบกลับ
print(response.json())

'''


'''
import json
import requests
from sentence_transformers import SentenceTransformer

# สร้าง SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ข้อความที่ต้องการแปลงเป็นเวกเตอร์
text = "four door"

# แปลงข้อความเป็นเวกเตอร์
vector = model.encode(text)

# ข้อมูลสำหรับการค้นหาใน Elasticsearch
url = "http://localhost:9200/my-index/_search"
headers = {'Content-Type': 'application/json'}

# สร้างคำขอสำหรับค้นหาด้วย KNN
data = {
    "size": 1,
    "query": {
        "knn": {
            "field": "vector",
            "query_vector": vector.tolist(),  # แปลงเวกเตอร์ให้เป็นรายการ
            "k": 3
        }
    }
}

# ส่งคำขอไปยัง Elasticsearch
response = requests.post(url, headers=headers, json=data, auth=('elastic', 'changeme'))

# แสดงผลลัพธ์
print(json.dumps(response.json(), indent=4))
'''

'''
import requests
from requests.auth import HTTPBasicAuth
import json

url = "http://localhost:9200/my-index/_search"
response = requests.get(url, auth=HTTPBasicAuth('elastic', 'changeme'))

# Check the response status and print the formatted result
if response.status_code == 200:
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Error: {response.status_code}")
'''



# yolo
'''
import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO('yolov8n.pt')  # ใช้โมเดล yolov8n.pt หรือสามารถเลือกโมเดลที่เหมาะสม

# โหลดภาพ
image_path = 'a1.jpg'  # ระบุเส้นทางภาพที่ถูกต้อง
image = cv2.imread(image_path)

# ตรวจสอบว่าภาพถูกโหลดสำเร็จหรือไม่
if image is None:
    print(f"ไม่สามารถโหลดภาพจาก {image_path}")
else:
    # ทำการตรวจจับวัตถุในภาพ
    results = model(image)

    # แสดงผลการตรวจจับ (ใช้ results[0] เพื่อแสดงผล)
    results[0].show()

    # ดึงข้อมูลจากผลการตรวจจับ
    labels = results[0].names  # รายชื่อวัตถุที่ตรวจจับได้
    # ผลลัพธ์ใน DataFrame
    results_df = results[0].to_df()  # แปลงผลลัพธ์เป็น DataFrame

    # ฟังก์ชันสำหรับการแยกสีจาก bounding box
    def extract_object_color(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)  # แปลงค่า bounding box เป็นตัวเลข
        roi = image[y1:y2, x1:x2]  # Crop ภาพใน bounding box
        avg_color = np.mean(roi, axis=(0, 1))  # หาค่าเฉลี่ยสี (BGR)
        return avg_color

    # สร้าง vector สำหรับเก็บข้อมูล
    object_data = []

    # อ่านข้อมูลจาก DataFrame
    for index, row in results_df.iterrows():

        bbox = [row['box']['x1'], row['box']['y1'], row['box']['x2'], row['box']['y2']]
        label = labels[int(row['class'])]
        color = extract_object_color(image, bbox)
        confidence = row['confidence']
        
        object_data.append({
            'label': label,
            'bounding_box': bbox,
            'average_color': color,
            'confidence': confidence
        })
        
    # แสดงข้อมูลที่เก็บ
    for data in object_data:
        print(data)

    # สามารถบันทึกข้อมูลใน vector database ที่ต้องการได้
'''

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

# สร้าง SentenceTransformer model
model_SentenceT = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# โหลดโมเดลและ processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_Blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# โหลดโมเดล YOLOv8
model = YOLO('yolov8n.pt')  # ใช้โมเดล yolov8n.pt หรือสามารถเลือกโมเดลที่เหมาะสม

# โหลดภาพ
image_path = 'a1.jpg'  # ระบุเส้นทางภาพที่ถูกต้อง
image = cv2.imread(image_path)


print(image.shape)

#################################################################################################################

def upload_image(bucket_name, object_name, image):

    global ip_minIO, name_key, password_key

    client = Minio(
                    ip_minIO,
                    access_key=name_key,
                    secret_key=password_key,
                    secure=False  # Or True, must use HTTPS
                  )
    try:
        if not client.bucket_exists(bucket_name):
            return

        success, image_buffer = cv2.imencode('.jpg', image)
        if not success:
            return
        image_bytes = BytesIO(image_buffer)
        image_bytes.seek(0)
        client.put_object(bucket_name, object_name, image_bytes, len(image_buffer))
    except S3Error as exc:
        print("Error occurred:", exc)
        pass

def upload_images_concurrently(bucket_name, image_files, image_names):
    with ThreadPoolExecutor() as executor:
        futures = []
        for image, name in zip(image_files, image_names):
            futures.append(executor.submit(upload_image, bucket_name, name, image))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                pass


time_capture_images = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

unique_id_str = "123456789"
name_json = "test"
image_name = f"{unique_id_str}_{name_json}.jpg"

ip_minIO="0.0.0.0:9100"
name_key="admin"
password_key="P@ssw0rd"
bucket_name="object"




def minio_upload(image, ip_minIO, name_key, password_key, bucket_name, image_name):

    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = io.BytesIO(buffer)

    # เชื่อมต่อ MinIO
    minio_client = Minio(
                            ip_minIO,
                            access_key=name_key,
                            secret_key=password_key,
                            secure=False  # เปลี่ยนเป็น True หากใช้ HTTPS
                        )

    # ตรวจสอบว่า bucket มีอยู่แล้วหรือไม่
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    # อัปโหลดไฟล์
    minio_client.put_object(
                                bucket_name,
                                image_name,
                                image_bytes,
                                length=len(buffer),
                                content_type="image/jpeg"
                            )

    link_image = "http://"+ip_minIO+"/"+bucket_name+"/"+image_name
    return link_image

link_image = minio_upload(image, ip_minIO, name_key, password_key, bucket_name, image_name)

#################################################################################################################

# ตรวจสอบว่าภาพถูกโหลดสำเร็จหรือไม่
if image is None:
    print(f"ไม่สามารถโหลดภาพจาก {image_path}")
else:
    # ทำการตรวจจับวัตถุในภาพ
    results = model(image)

    # แสดงผลการตรวจจับ (ใช้ results[0] เพื่อแสดงผล)
    results[0].show()

    # ดึงข้อมูลจากผลการตรวจจับ
    labels = results[0].names  # รายชื่อวัตถุที่ตรวจจับได้
    # ผลลัพธ์ใน DataFrame
    results_df = results[0].to_df()  # แปลงผลลัพธ์เป็น DataFrame

    # ฟังก์ชันสำหรับการแยกสีจาก bounding box
    def extract_object_color(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)  # แปลงค่า bounding box เป็นตัวเลข
        roi = image[y1:y2, x1:x2]  # Crop ภาพใน bounding box

        # Set the quality parameter to 50 (lower value means higher compression, and lower quality)
        roi_base64 = base64.b64encode(cv2.imencode('.jpeg', roi)[1]).decode()

        image_data_base64 = base64.b64encode(cv2.imencode('.jpeg', roi,[cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
        roi_base64 = (f"data:image/jpeg;base64,{image_data_base64}")

        inputs = processor(images=roi, return_tensors="pt")
        out = model_Blip.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption, roi_base64

    # สร้าง vector สำหรับเก็บข้อมูล
    object_data = []

    # อ่านข้อมูลจาก DataFrame
    for index, row in results_df.iterrows():

        #bbox = [row['box']['x1'], row['box']['y1'], row['box']['x2'], row['box']['y2']]
        bbox = list(map(int, [row['box']['x1'], row['box']['y1'], row['box']['x2'], row['box']['y2']]))

        label = labels[int(row['class'])]
        caption, roi_base64 = extract_object_color(image, bbox)

        confidence = row['confidence']
        
        text = str(label) + " " + str(caption)        
        # แปลงข้อความเป็นเวกเตอร์
        vector = model_SentenceT.encode(text)

        time_stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # "image": bbox
        # "vector": vector.tolist(),  # เปลี่ยนเวกเตอร์เป็น list เพื่อให้ส่งได้
        data = {
            "vector":"sssss",  # เปลี่ยนเวกเตอร์เป็น list เพื่อให้ส่งได้
            "text": text, 
            "bbox": bbox, 
            "urlImage" : link_image,
            "timeCaptureImage" : time_capture_images, 
            "timeStamp" : time_stamp

        }

        # สร้าง ID ใหม่ด้วย uuid เพื่อหลีกเลี่ยงการทับข้อมูลเดิม
        document_id = str(uuid.uuid4())  # ใช้ UUID สำหรับ ID ที่ไม่ซ้ำ

        # ส่งข้อมูลไปยัง Elasticsearch
        url = f"http://localhost:9200/my-index/_doc/{document_id}"
        headers = {'Content-Type': 'application/json'}
        auth = ('elastic', 'changeme')

        # ส่ง POST request
        #response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))

        # แสดงผลการตอบกลับ
        #print(response.json())

        print(data)

        '''
        object_data.append({
            'label': label,
            'bounding_box': bbox,
            'caption': str(label +" "+ caption),
            'confidence': confidence
        })
        '''
        
    # แสดงข้อมูลที่เก็บ
    #for data in object_data:
    #    print(data)

    # สามารถบันทึกข้อมูลใน vector database ที่ต้องการได้




'''
# สร้าง SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ข้อความที่ต้องการแปลงเป็นเวกเตอร์
text = "two-door sedan"

# แปลงข้อความเป็นเวกเตอร์
vector = model.encode(text)

# สร้างข้อมูลที่ต้องการส่งไปยัง Elasticsearch
data = {
    "vector": vector.tolist(),  # เปลี่ยนเวกเตอร์เป็น list เพื่อให้ส่งได้
    "text": text
}

# สร้าง ID ใหม่ด้วย uuid เพื่อหลีกเลี่ยงการทับข้อมูลเดิม
document_id = str(uuid.uuid4())  # ใช้ UUID สำหรับ ID ที่ไม่ซ้ำ

# ส่งข้อมูลไปยัง Elasticsearch
url = f"http://localhost:9200/my-index/_doc/{document_id}"
headers = {'Content-Type': 'application/json'}
auth = ('elastic', 'changeme')

# ส่ง POST request
response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))

# แสดงผลการตอบกลับ
print(response.json())
'''