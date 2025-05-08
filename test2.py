import requests
from requests.auth import HTTPBasicAuth
import json

'''
url = "http://localhost:9200/my-index/_delete_by_query"
data = {
    "query": {
        "match_all": {}
    }
}

response = requests.post(url, json=data, auth=HTTPBasicAuth('elastic', 'changeme'))

# Check the response status and print the result
if response.status_code == 200:
    print("All documents deleted successfully.")
else:
    print(f"Error: {response.status_code}, {response.text}")
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

def search_data_by_id(url_database, name_database, doc_id):
    try:
        es_url = url_database + "/" + name_database + "/_doc/" + doc_id
        
        auth = ("elastic", "changeme")  # ใส่ username และ password ของ Elasticsearch
        headers = {"Content-Type": "application/json"}

        # Perform the search request to get the document by _id
        response = requests.get(es_url, auth=auth, headers=headers)

        #print(response.status_code)  
        if response.status_code == 200:
            #print(response.json())  # Output the document content
            #print(json.dumps(response.json(), indent=4))
            result = response.json()
        else:
            result = f"Error: {response.status_code}"

    except Exception as e:
        #print(f"Error: {e}")
        result = f"Error: {e}"

    return result


import requests
from requests.auth import HTTPBasicAuth
import json
import cv2
import urllib.request
import numpy as np

def extract_object_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)  # แปลงค่า bounding box เป็นตัวเลข
    roi = image[y1:y2, x1:x2]  # Crop ภาพใน bounding box

    return roi

url = "http://localhost:9200/my-vertor/_search"
response = requests.get(url, auth=HTTPBasicAuth('elastic', 'changeme'))

link_imges_ref = None

# Check the response status and print the formatted result
if response.status_code == 200:
    # Extracting only the "id_database" from each hit
    hits = response.json().get('hits', {}).get('hits', [])
    for hit in hits:        

        id_database = hit['_source'].get('id_database')

        url_database = "http://0.0.0.0:9200"
        name_database = "my-databaes-general"

        result = search_data_by_id(url_database, name_database, id_database)
        link_imges = result['_source'].get('urlImage')
        location = result['_source'].get('location')
        timeCaptureImage = result['_source'].get('timeCaptureImage')
        timeStamp = result['_source'].get('timeStamp')

        print(
            hit['_source'].get('id_database'), 
            hit['_source'].get('text'),
            hit['_source'].get('bbox'),
            hit['_source'].get('time_stamp'),
            link_imges,
            location,
            timeCaptureImage,
            timeStamp
            )
        
        print(link_imges_ref, id_database)

        if link_imges_ref == None:
            link_imges_ref = id_database

            resp = urllib.request.urlopen(link_imges)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)

            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            print(image.shape)          

        if link_imges_ref != id_database :

            resp = urllib.request.urlopen(link_imges)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)

            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            print(image.shape)

else:
    print(f"Error: {response.status_code}")



