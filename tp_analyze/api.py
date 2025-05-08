from fastapi import FastAPI
from fastapi.responses import  JSONResponse
import json
import time
import sys
import os
import os.path
from gevent.pywsgi import WSGIServer
import signal
import configparser
from datetime import datetime
import socket
from network_add import networks_set
import importlib.util

import uvicorn
from pydantic import BaseModel

from script_add import ai_status, check_connection, file_status
from change_method import change
try:
    change()
except configparser.Error as e:
    print(f"Change type ai error: {e}")
    sys.exit(1)
import glob
path_method = configparser.ConfigParser()
path_method.read('config_set_up_api_analyze.txt')
type_load_model = path_method['parameters-set-up'].get('method_analyze', None)
sys.path.append('./load_model/'+type_load_model+'/')

from load_model import main_load
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

hostname = (socket.gethostname())
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

app = FastAPI()

################# Check status strat #################
@app.get("/statusAnalyze")
async def status_ai():
    return {"message": "GET request successful", "status": "OK"}, 200
################# Check status end #################

class requestInput(BaseModel):
    urlSend: str
    timeStamp: str
    nameFile: str
    jobNumber: str
    hwId: str
    workPlanId: str
    dataDerail: list
    images : str
    aiType: str

@app.post("/analyze", response_class=JSONResponse)
async def predict(req_dara: requestInput):

    global load_model_all

    dt_time_record = datetime.now()
    time_record =  dt_time_record.strftime("%Y-%m-%dT%H:%M:%SZ")

    start_time = time.time()

    try:
        start_time = time.time()
        #data_req = req_dara.dict()
        data_req = req_dara.model_dump()
        data_req = {key: value for key, value in data_req.items() if value != 'null'}
        data = json.dumps(data_req, ensure_ascii=False)  # แปลงเป็น JSON

        obj = loaded_method[method_analyze].AnalyzeAPI(data, load_model_all, start_time)        
        json_output = obj.analyze_image()
        #json_output["timeRecord"] = time_record
        json_output = "aaa"
    except Exception as e:
        print(f"Error: {e}")
        json_output =  {
                         "cause": f"The submitted data format is invalid., error: {e}"                
                      }
        json_output["timeRecord"] = time_record
        pass
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time : {elapsed_time:.4f} seconds")
    print("")
    return json_output

def stop_server(*args):
    global running
    running = False
    if http_server is not None:
        http_server.stop()
    try:
        file_status("status_health", "main-health", "status_health", False)
    except configparser.Error as e:
        print(f"save file of status health Error: {e}")
        pass
    print("Server stopped.")

def show_config(start_ime, path_config):
    with open(path_config, 'r') as file:
        config_data = file.read()
    result = (
        "_________Analyze (POINT IT Consulting)__________\n"
        f"Start ime : {start_ime}\n"
        "-----------------Config API analyze-----------------\n"
        f"{config_data}"
        "_________________________________________________"
    )
    print(result)

if __name__ == "__main__":

    try:
        file_status("status_health", "main-health", "status_health", False)
    except configparser.Error as e:
        print(f"save file of status health Error: {e}")
        sys.exit(1)

    try:
        networks_set()
    except configparser.Error as e:
        print(f"ConfigParser Error: {e}")
        sys.exit(1)

    running = True

    startTime = datetime.now()
    startTime =  startTime.strftime("%Y-%m-%dT%H:%M:%SZ")

    config_set_up = configparser.ConfigParser()
    config_set_up.read('config_set_up_api_analyze.txt')

    api_url = config_set_up['parameters-set-up'].get('api_url', None)
    port = config_set_up['parameters-set-up'].getint('port', None)
    list_model = config_set_up['parameters-set-up'].get('list_model', '').split(',')

    nameOutput = config_set_up['parameters-set-up'].get('folder_name', None)
    method_analyze = config_set_up['parameters-set-up'].get('method_analyze', None)
    file_name_config = config_set_up['parameters-set-up'].get('file_name_config', None)

    url_leader = config_set_up['send-status'].get('url_leader', None)
    port_leader = config_set_up['send-status'].getint('port_leader', None)
    send_leader = f"http://{url_leader}:{port_leader}/statusAi"

    ###### check_connection start ######
    status_leader = f"http://{url_leader}:{port_leader}/statusLeader"
    #check_connection(status_leader)
    ###### check_connection end ######

    show_config(startTime, file_name_config)       

    folder_name = './'+nameOutput
    check_folder = os.path.exists(folder_name)
    if check_folder == False :
        print("not found folder : create folder...")
        os.mkdir(folder_name)
        name_folder = 'images_'
        mark = '/'
        for i in range(10):
            folders_to_create = name_folder+str(i+1)
            os.mkdir(folder_name+mark+folders_to_create)
        os.mkdir(folder_name+mark+'output_json')

    load_model_all = main_load(list_model)
    loaded_method = {}
    
    method_files = glob.glob('./method_analyze/'+method_analyze+'/method*.py')
    if method_files == [] :
        print("Can't find the file!!!") 
        sys.exit(0)
    else :
        for method in method_files:
            module_name = os.path.splitext(os.path.basename(method))[0]
            spec = importlib.util.spec_from_file_location(module_name, method)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded_method[module_name] = module
    
    '''
    try :
        dt_checker = datetime.now()
        time_checker =  dt_checker.strftime("%Y-%m-%dT%H:%M:%SZ")
        ai_status(send_leader, "ready", hostname)
        print(f"time checker: {time_checker} Request successful connect leader")
    except :
        print(f"time checker: {time_checker} An error occurred, cannot connect leader")
        sys.exit(1)
    '''

    signal.signal(signal.SIGINT, stop_server) 
    http_server = WSGIServer((api_url, port), app)
    print("Server started on port :", port)
    try:
        file_status("status_health", "main-health", "status_health", True)
    except configparser.Error as e:
        print(f"save file of status health Error: {e}")
        sys.exit(1)
    try:
        uvicorn.run(app, host=api_url, port=port)
    except KeyboardInterrupt:
        stop_server()
