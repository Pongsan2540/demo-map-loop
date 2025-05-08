import requests
import json
from datetime import datetime

es_url = "http://localhost:9200/my-databaes-general"
auth = ("elastic", "changeme")  # ใส่ username และ password ของ Elasticsearch
headers = {"Content-Type": "application/json"}

data = {
    "mappings": {
        "properties": {
            "timeStamp": {"type": "text"},
            "timeCaptureImage": {"type": "text"},
            "urlImage": {"type": "text"},
            "address" : {"type": "text"}
        }
    }
}

response = requests.put(es_url, auth=auth, headers=headers, data=json.dumps(data))

print(response.status_code)  
print(response.json())  



'''
import requests
import json

# ตั้งค่าตัวแปร
es_url = "http://localhost:9200/my-index-database/_doc"  # ไม่กำหนด ID เพื่อให้ Elasticsearch สร้างให้เอง
auth = ("elastic", "changeme")
headers = {"Content-Type": "application/json"}

# ข้อมูลที่ต้องการเพิ่ม
data = {
    "name": "zeen ponn",
    "age": 20,
    "created_at": "2024-04-01T12:00:00",
    "metadata": {
        "city": "all",
        "job": "cddd"
    }
}

# ส่งคำขอ POST เพื่อเพิ่มข้อมูล
response = requests.post(es_url, auth=auth, headers=headers, data=json.dumps(data))

# ตรวจสอบผลลัพธ์
if response.status_code == 201:
    response_data = response.json()
    doc_id = response_data["_id"]  # ดึงค่า _id
    print(f"เพิ่มข้อมูลสำเร็จ! _id: {doc_id}")
else:
    print(f"เกิดข้อผิดพลาด: {response.json()}")

'''


'''
import requests
from requests.auth import HTTPBasicAuth

# กำหนด URL ของ Elasticsearch
url = "http://localhost:9200/my-index-database"

# ส่งคำขอ DELETE ไปยัง Elasticsearch
response = requests.delete(url, auth=HTTPBasicAuth("elastic", "changeme"))

# แสดงผลลัพธ์
if response.status_code == 200:
    print("Index deleted successfully")
else:
    print(f"Failed to delete index: {response.status_code}")
    print(response.text)
'''