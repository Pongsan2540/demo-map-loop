import requests
import time

def ai_status(send_leader, status_detect, hostname):    
    
    final_status = f"{status_detect}-{hostname}"
    ai_status = {
                "statusDetection": "null",
                "statusCreate": "null",
                "statusManage": "null",
                "statusAnalyze": final_status,
                "statusResult": "null",
                }

    response_ai_status = requests.post(send_leader, json=ai_status, timeout=3)
    response_ai_status.raise_for_status()  # Raise an exception if the request was not successful

def check_connection(url, retry_interval=1):
    
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Successfully connected to {url}")
                break
            else:
                print(f"Failed to connect to {url}, Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Unable to connect to {url}: {e}")
            print("Reconnecting ...")
        time.sleep(retry_interval)

def file_status(name_file, main_topic, sub_topic, status):
    content = f"""[{main_topic}]\n{sub_topic}={status}"""
    filename = f"{name_file}.txt"
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Content saved to {filename}")
