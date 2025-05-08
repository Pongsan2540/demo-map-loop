import requests
import json
import requests
from requests.auth import HTTPBasicAuth
import argparse

def create_database(url_database, name_database):
    try: 
        es_url = url_database + "/" + name_database
        
        print(es_url)

        auth = ("elastic", "changeme")  # ใส่ username และ password ของ Elasticsearch
        headers = {"Content-Type": "application/json"}

        data = {
            "mappings": {
                "properties": {
                    "timeStamp": {"type": "text"},
                    "timeCaptureImage": {"type": "text"},
                    "urlImage": {"type": "text"},
                    "location" : {"type": "text"}
                }
            }
        }

        response = requests.put(es_url, auth=auth, headers=headers, data=json.dumps(data))

        print(response.status_code)  
        print(response.json())  
        result = response.status_code

    except Exception as e:
        print(f"Error: {e}")
        result = f"Error: {e}"

    return result

def delete_database(url_database, name_database):

    try:
        url = url_database + "/" + name_database
        response = requests.delete(url, auth=HTTPBasicAuth("elastic", "changeme"))

        # แสดงผลลัพธ์
        if response.status_code == 200:
            print("Index deleted successfully")
            result = "Index deleted successfully"
        else:
            print(f"Failed to delete index: {response.status_code}")
            print(response.text)
        result = response.status_code

    except Exception as e:
        print(f"Error: {e}")
        result = f"Error: {e}"

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api")
    parser.add_argument('--name_database', default='my-databaes-general', help="my-databaes-general")
    parser.add_argument('--url_database', default='http://localhost:9200', help="http://localhost:9200")
    parser.add_argument('--type_method', default='create', help="create or delete")

    args = parser.parse_args()

    name_database = args.name_database 
    url_database = args.url_database 
    type_method = args.type_method 

    delete_database(url_database, name_database)
    create_database(url_database, name_database)

    #if type_method == "create" :
    #    create_database(url_database, name_database)
    #elif type_method == "delete" :
    #    delete_database(url_database, name_database)
    #else :
    #    print("Choosing the wrong method")

