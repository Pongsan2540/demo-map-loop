import configparser
import concurrent.futures
from datetime import datetime
import requests
import json
import numpy as np
import time

import base64
import json
import cv2
import torch
import os.path
import sys
import uuid

import folium
import base64

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(script_dir, 'script.py')
sys.path.append(os.path.dirname(script_dir))
from script import minio_upload, extract_object_color

headers = {"Content-Type": "application/json; charset=utf-8"}
########## config bypass start ###########
config_bypass = configparser.ConfigParser()
config_bypass.read('config_bypass.txt')
by_pass_url = config_bypass['main-bypass'].get('by_pass_url', None)
by_pass_port = config_bypass['main-bypass'].get('by_pass_port', None)
by_pass_name = config_bypass['main-bypass'].get('by_pass_name', None)
########## config bypass end ###########

########## config setup start ###########
config_set_up = configparser.ConfigParser()
config_set_up.read("config_set_up_api_analyze.txt")
save_img_all = config_set_up['parameters-set-up'].getint('save_img', None)
type_analsze_lsq = config_set_up['parameters-set-up'].getint('type_analsze', None)
folder_name = './'+config_set_up['parameters-set-up'].get('folder_name', None)
method_analyze = config_set_up['parameters-set-up'].get('method_analyze', None)
save_img_result = 0
save_json = 0
########## config setup end ###########

########## config tempesta start ###########
config_tempesta = configparser.ConfigParser()
config_tempesta.read('./config_tempesta/config_tempesta.conf')
type_ai = config_tempesta['parameters-set-up'].get('aitype', None)
type_ai_list = str(type_ai.lower().split()[0])
########## config setup end ###########

########## config method start ###########
config_method = configparser.ConfigParser()
config_method.read('./method_analyze/'+str(method_analyze)+'/config_method.txt')
cal_plxels = config_method['config-method'].getfloat('one_plxel_one_mm', None)
########## config method end ###########

class AnalyzeAPI:

    def __init__(self, data, load_model_all, start_time):
        
        model_sentence, model_blip, model_yolo, device= load_model_all

        self.model_sentence = model_sentence
        self.model_blip = model_blip
        self.model_yolo  = model_yolo  
        self.device = device 
        self.json_data = json.loads(data)
        self.start_time = start_time

    def analyze_image(self):
        
        def get_coordinates(address):
            api_key = "31feb3e1e4ad4789a2427ff627d145ab"  # แทนที่ด้วย API Key ของคุณ
            base_url = f"https://api.opencagedata.com/geocode/v1/json"

            params = {"q": address, "key": api_key}

            try:
                response = requests.get(base_url, params=params, timeout=10)
                data = response.json()

                if response.status_code != 200:
                    print(f"ข้อผิดพลาด HTTP: {response.status_code}, รายละเอียด: {data}")
                    return None

                if data["results"]:
                    location = data["results"][0]["geometry"]
                    return location["lat"], location["lng"]
                else:
                    print("ไม่พบพิกัดที่ตรงกับที่อยู่")
                    return None

            except requests.exceptions.RequestException as e:
                print(f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
                return None

        url_output = self.json_data.get("URL_SEND", None)
        time_stamp = self.json_data.get("TIME_STAMP", None)
        name_file = self.json_data.get("NAME_FILE", None)
        jobNumber = self.json_data.get("JOB_NUMBER", None)
        hwId = self.json_data.get("HW_ID", None)
        workplanId = self.json_data.get("WORKPLAN_ID", None)
        data_detail = self.json_data.get("DATA_DERAIL", None)
        aiType = self.json_data.get("AI_TYPE", None)


        model_sentence = self.model_sentence
        model_blip = self.model_blip
        model_yolo = self.model_yolo
        device = self.device

        if url_output is None:
            url_output = "http://"+by_pass_url+":"+by_pass_port+"/"+by_pass_name
            aiType = type_ai_list
        else:
            pass

        images_value = self.json_data.get("IMAGE", None)

        try: 

            print("zzz")
              
            time_capture_images = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            es_url = "http://localhost:9200/my-databaes-general/_doc"  # ไม่กำหนด ID เพื่อให้ Elasticsearch สร้างให้เอง
            auth = ("elastic", "changeme")
            headers = {"Content-Type": "application/json"}

            import random
            import string
            def random_string(length=10):
                return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

            data = {    
                        "timeStamp": datetime.now().isoformat(),
                        "timeCaptureImage": "2025-04-01T17:29:40.046956",
                        "urlImage": "http://0.0.0.0:9100/object/123456789_test.jpg",
                        "nameCam":"โรงพยาบาลสมิติเวช สุขุมวิท",
                        "typeCam":"test Cam",
                        "location": "133 สุขุมวิท 49 แขวงคลองตันเหนือ เขตวัฒนา กรุงเทพมหานคร 10110"
                    }
            
            location = data["location"]


            coordinates = get_coordinates(location)
            print(f"พิกัด GPS: {coordinates[0]}, {coordinates[1]}")

            gps_location = (coordinates[0], coordinates[1]) 

            
            img = cv2.imread('/home/pointit/Documents/milvus/a1.jpg')
            _, buffer = cv2.imencode('.jpg', img)
            encoded_string = base64.b64encode(buffer).decode("utf-8")

            popup_html = f"""
                <div style="text-align:center;">
                    <h4>📍 จุดกล้อง: โรงพยาบาลสมิติเวช สุขุมวิท</h4>
                    <img src="data:image/jpeg;base64,{encoded_string}" width="250px"><br>
                    <p>133 สุขุมวิท 49 แขวงคลองตันเหนือ เขตวัฒนา กรุงเทพมหานคร 10110</p>
                </div>
            """

            m = folium.Map(location=gps_location, zoom_start=15)

            folium.Marker(
                location=gps_location,
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="red", icon="glyphicon glyphicon-map-marker"),
            ).add_to(m)

            m.save("map_with_image.html")
            print("✅ บันทึกแผนที่สำเร็จ: map_with_image.html")
            

            data["gps"] = coordinates
            data["map"] = "http://0.0.0.0:9100/map/123456789_test.html"

            #print(data)
         
            response = requests.post(es_url, auth=auth, headers=headers, data=json.dumps(data))

            if response.status_code == 201:
                response_data = response.json()
                doc_id = response_data["_id"]  # ดึงค่า _id
                print(f"เพิ่มข้อมูลสำเร็จ! _id: {doc_id}")
            else:
                print(f"เกิดข้อผิดพลาด: {response.json()}")
                doc_id = False


            image_path = '/home/pointit/Documents/milvus/a1.jpg'  # ระบุเส้นทางภาพที่ถูกต้อง
            image = cv2.imread(image_path)

            time_capture_images = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            unique_id_str = "123456789"
            name_json = "test"
            image_name = f"{unique_id_str}_{name_json}.jpg"

            ip_minIO="0.0.0.0:9100"
            name_key="admin"
            password_key="P@ssw0rd"
            bucket_name="object"

            #link_image = minio_upload(image, ip_minIO, name_key, password_key, bucket_name, image_name)
            link_image = "12345"

            if image is None:
                print(f"ไม่สามารถโหลดภาพจาก {image_path}")
            else:
                results = model_yolo(image)

                labels = results[0].names  # รายชื่อวัตถุที่ตรวจจับได้
                results_df = results[0].to_df()  # แปลงผลลัพธ์เป็น DataFrame

                object_data = []

                for index, row in results_df.iterrows():

                    bbox = list(map(int, [row['box']['x1'], row['box']['y1'], row['box']['x2'], row['box']['y2']]))
                    label = labels[int(row['class'])]

                    caption = extract_object_color(image, model_blip, bbox)

                    confidence = row['confidence']
                    
                    text = str(label) + " " + str(caption)        
                    vector = model_sentence.encode(text)

                    time_stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

                    data = {
                            "id_database": doc_id,
                            "vector": vector.tolist(),  
                            "text": text, 
                            "bbox": bbox, 
                            "timeStamp" : time_stamp,
                            "label" : label
                        }


                    document_id = str(uuid.uuid4())  # ใช้ UUID สำหรับ ID ที่ไม่ซ้ำ

                    url = f"http://localhost:9200/my-vertor/_doc/{document_id}"
                    headers = {'Content-Type': 'application/json'}
                    auth = ('elastic', 'changeme')

                    response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))

                    #print(json.dumps(response.json(), indent=4))

            statusData = True

        except Exception as e:
            statusData = f"Unable to analyze image, error: {e}"
            print(statusData)
            pass

        return statusData