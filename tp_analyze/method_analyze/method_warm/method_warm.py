import configparser
import concurrent.futures
from datetime import datetime
import requests
import json
import numpy as np
import re
import time

import base64
import json
import cv2
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms as T
from segment_anything import sam_model_registry, SamPredictor
import os.path
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(script_dir, 'script.py')
sys.path.append(os.path.dirname(script_dir))
from script import analyze_lsq_A, analyze_lsq_B_1, analyze_lsq_B_2, analyze_lsq_B_3, analyze_lsq_B_4, images_equalizeHist, rectangle_images, overlay_image, move_points

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
        
        model1, model2, model3, device = load_model_all

        self.model_unet_1 = model1
        self.model_unet_2 = model2
        self.predictor_sam  = model3  
        self.device = device 
        self.json_data = json.loads(data)
        self.start_time = start_time

    def analyze_image(self):

        def test_predict_image_mask_miou(model, image, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            model.eval()
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            image = t(image)
            model.to(device)
            image = image.to(device)

            with torch.no_grad():
                image = image.unsqueeze(0)
                output = model(image)
                masked = torch.argmax(output, dim=1)
                masked = masked.cpu().squeeze(0)
            return masked
        

        def B2_images(predictor, images, image_resize):
            image = images    
            h, w,_ = image.shape
            try:
                center_hight = int(h/2)
                data_position = np.where(image_resize[center_hight, :] == 255)[0]
                number_data = len(data_position)+5

                x1, y1 = min(data_position)-number_data/2, center_hight-60
                x2, y2 = max(data_position)+number_data/2, center_hight+60

                input_box = np.array([x1, y1, x2, y2])

                predictor.set_image(image)

                masks, scores, _ = predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box[None, :],
                                    multimask_output=False,
                                    )

                max_scores = max(scores)
                list_max_mask = list(scores).index(max_scores)

                masks_best = masks[list_max_mask]
                my_array = np.where(masks_best == False, 0, masks_best)
                my_array = np.where(my_array == True, 255, my_array)
                my_array = my_array.astype(np.uint8)

                kernel = np.ones((5,5),np.uint8)
                opening = cv2.morphologyEx(my_array, cv2.MORPH_OPEN, kernel)
                image_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                
                contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                coordinate_position = [cv2.boundingRect(cnt) for cnt in contours]
                
                max_hight_index = max(enumerate(coordinate_position), key=lambda x: x[1][-1])[0]
                position_value = coordinate_position[max_hight_index]    
                    
                _, _, _, hight_max = position_value

            except:
                hight_max = 100
            return hight_max
        
        def json_save(name_file, folder_name, json_data):

            json_object = json.dumps(json_data, indent=3)
            with open(folder_name+'/'+'output_json'+'/'+str(name_file)+'.json', 'w') as outfile:
                outfile.write(json_object)
            return json_object
        
                                    
        url_output = self.json_data.get("URL_SEND", None)
        time_stamp = self.json_data.get("TIME_STAMP", None)
        name_file = self.json_data.get("NAME_FILE", None)
        jobNumber = self.json_data.get("JOB_NUMBER", None)
        hwId = self.json_data.get("HW_ID", None)
        workplanId = self.json_data.get("WORKPLAN_ID", None)
        data_detail = self.json_data.get("DATA_DERAIL", None)
        aiType = self.json_data.get("AI_TYPE", None)


        predictor_sam = self.predictor_sam
        model_unet = self.model_unet_1
        model_unet_2 = self.model_unet_2
        device = self.device

        if url_output is None:
            url_output = "http://"+by_pass_url+":"+by_pass_port+"/"+by_pass_name
            aiType = type_ai_list
        else:
            pass

        if isinstance(data_detail, list) and data_detail:
            index_butt = [item for item in data_detail if isinstance(item, list) and item[0] == 'pig_butt'][0]
            index_loin = [item for item in data_detail if isinstance(item, list) and item[0] == 'pig_loin'][0]
            index_carcass = [item for item in data_detail if isinstance(item, list) and item[0] == 'carcass'][0]
            pass
        else:
            index_butt = None
            index_loin = None
            index_carcass = None

        images_value = self.json_data.get("IMAGE", None)

        try: 

            name_butt , b_top_butt, b_left_butt, b_width_butt, b_height_butt, pro_butt = index_butt
            name_loin , b_top_loin, b_left_loin, b_width_loin, b_height_loin, pro_loin = index_loin
            name_carcass , b_top_carcass, b_left_carcass, b_width_carcass, b_height_carcass, pro_carss = index_carcass
            
            index_butt = [name_butt, b_left_butt, b_top_butt, b_left_butt + b_width_butt, b_top_butt + b_height_butt, pro_butt]
            index_loin = [name_loin,   b_left_loin, b_top_loin, b_left_loin + b_width_loin, b_top_loin + b_height_loin, pro_loin]
            index_carcass = [name_carcass,  b_left_carcass, b_top_carcass, b_left_carcass + b_width_carcass, b_top_carcass + b_height_carcass, pro_carss]

            _ , x1_butt, y1_butt, x2_butt, y2_butt, _= index_butt
            _ , x1_loin, y1_loin, x2_loin, y2_loin, _ =  index_loin
            _, x1_carcass, y1_carcass, x2_carcass, y2_carcass, _ = index_carcass

            pattern = rb'data:image/jpeg;base64,'
            images_value = re.sub(pattern, b'', images_value.encode())  # Convert to bytes before substitution
            image_array = np.frombuffer(base64.b64decode(images_value), dtype=np.uint8)
            image_full = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            image_nocrop_data_base64 = base64.b64encode(cv2.imencode('.jpeg', image_full,
                                            [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            data_nocrop_base64 = (f"data:image/jpeg;base64,{image_nocrop_data_base64}")

            image = image_full[y1_carcass:y2_carcass, x1_carcass:x2_carcass]

            image_og = image.copy()

            h, w,_ = image.shape
            d_h = int(h/3)
            input_point = np.array([[int(w/2), int((h/2)-d_h)], [int(w/2), int(h/2)], [int(w/2), int((h/2)+d_h)]])
            input_label = np.array([1, 1, 1])


            predictor_sam.set_image(image)
            masks, scores, logits = predictor_sam.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,
                                )
            
            max_scores = max(scores)
            list_max_mask = list(scores).index(max_scores)

            masks_best = masks[list_max_mask]
            my_array = np.where(masks_best == False, 0, masks_best)
            my_array = np.where(my_array == True, 255, my_array)
            my_array = my_array.astype(np.uint8)

            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(my_array, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            lengthy_array = [len(c) for c in contours]
            position_array = [cv2.boundingRect(c) for c in contours]
            max_index, max_position = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])
        
            mask_image = np.zeros_like(closing)
            cv2.drawContours(mask_image, contours, max_index, (255,255,255), -1)

            images_result = cv2.bitwise_and(image_og, image_og, mask=mask_image)
            images_histogram = images_equalizeHist(images_result)

            _ , x1_butt, y1_butt, x2_butt, y2_butt, _ = index_butt
            _ , x1_loin, y1_loin, x2_loin, y2_loin, _ =  index_loin
            _, x1_carcass, y1_carcass, x2_carcass, y2_carcass, _ = index_carcass

            top_butt, left_butt, width_butt, height_butt = y1_butt, x1_butt, abs(x2_butt - x1_butt), abs(y2_butt - y1_butt)
            top_loin, left_loin, width_loin, height_loin =  y1_loin, x1_loin, abs(x2_loin - x1_loin), abs(y2_loin - y1_loin)
            top_carcass, left_carcass, width_carcass, height_carcass = y1_carcass, x1_carcass, abs(x2_carcass - x1_carcass), abs(y2_carcass - y1_carcass)

            y2_butt = (left_butt + width_butt)
            y2_loin = (left_loin + width_loin)
            y2_carcass = (left_carcass + width_carcass)

            if y2_butt > y2_carcass :
                width_butt = abs(width_butt-abs(y2_butt-y2_carcass))
            elif y2_loin > y2_carcass :
                width_loin = abs(width_loin-abs(y2_loin-y2_carcass))
            elif left_butt <  left_carcass :
                left_butt = left_carcass
                width_butt = abs(width_butt-abs(left_carcass-left_butt))
            elif left_loin < left_carcass :
                left_loin = left_carcass
                width_loin = abs(width_loin-abs(left_carcass-left_loin))


            if (top_loin+height_loin) > (top_carcass+height_carcass) :
                height_loin = abs(height_loin - abs((top_loin+height_loin) - (top_carcass+height_carcass)))

            move_x_carcass, move_y_carcass = 0, 0
            move_x_butt, move_y_butt = move_points(top_carcass, left_carcass, top_butt, left_butt, move_x_carcass, move_y_carcass)
            move_x_loin, move_y_loin = move_points(top_carcass, left_carcass, top_loin, left_loin, move_x_carcass, move_y_carcass)

            center_images = left_carcass + (width_carcass/2)
            if left_butt + (width_butt/2) > center_images :
                side = "R"
            elif left_butt + (width_butt/2) < center_images :
                side = "L"
            else :
                side = "Null"

            images_og_c1 = images_result[ move_x_butt : move_x_butt + height_butt, move_y_butt : move_y_butt + width_butt ]
            images_og_c2 = images_result[ move_x_loin : move_x_loin + height_loin, move_y_loin : move_y_loin + width_loin ]

            images_yolo_c1 = images_histogram[ move_x_butt : move_x_butt + height_butt, move_y_butt : move_y_butt + width_butt ]
            images_yolo_c2 = images_histogram[ move_x_loin : move_x_loin + height_loin, move_y_loin : move_y_loin + width_loin ]

            mask_image_c2 = mask_image[ move_x_loin : move_x_loin + height_loin, move_y_loin : move_y_loin + width_loin ]

            images_result = images_result.copy()
            h_og, w_og, _ = images_result.shape
            dsize = (1400,3650)
            images_size = cv2.resize(images_result, dsize)

            h_A, w_A, _ = images_yolo_c1.shape
            h_B, w_B, _ = images_yolo_c2.shape

            images_yolo_c1_RGB = cv2.cvtColor(images_yolo_c1, cv2.COLOR_BGR2RGB)
            images_unet_c1 = cv2.resize(images_yolo_c1_RGB, (1152, 768), interpolation=cv2.INTER_NEAREST)
            pred_mask_A = test_predict_image_mask_miou(model_unet, images_unet_c1, device)

            img_prediction_A = np.array(pred_mask_A)

            imgA_class_1 = np.where(img_prediction_A == 1, 255, img_prediction_A)
            imgA_class_2 = np.where(img_prediction_A == 2, 255, img_prediction_A)

            imgA_class_1 = np.where(imgA_class_1 == 2, 0, imgA_class_1)
            imgA_class_2 = np.where(imgA_class_2 == 1, 0, imgA_class_2)

            images_yolo_c2_RGB = cv2.cvtColor(images_yolo_c2, cv2.COLOR_BGR2RGB)
            images_unet_c2 = cv2.resize(images_yolo_c2_RGB, (1152, 768), interpolation=cv2.INTER_NEAREST)
            pred_mask_B = test_predict_image_mask_miou(model_unet, images_unet_c2, device)

            img_prediction_B = np.array(pred_mask_B)

            imgB_class_1 = np.where(img_prediction_B == 3, 255, img_prediction_B)
            imgB_class_2 = np.where(img_prediction_B == 4, 255, img_prediction_B)

            imgB_class_1 = np.where(imgB_class_1 == 4, 0, imgB_class_1)
            imgB_class_2 = np.where(imgB_class_2 == 3, 0, imgB_class_2)

            imgA_class_1 = cv2.resize(imgA_class_1, (w_A, h_A), interpolation=cv2.INTER_NEAREST)
            imgA_class_2 = cv2.resize(imgA_class_2, (w_A, h_A), interpolation=cv2.INTER_NEAREST)

            imgB_class_1 = cv2.resize(imgB_class_1, (w_B, h_B), interpolation=cv2.INTER_NEAREST)
            imgB_class_2 = cv2.resize(imgB_class_2, (w_B, h_B), interpolation=cv2.INTER_NEAREST)

            ###################################### start class 5 b3
            
            images_histogram_RGB = cv2.cvtColor(images_histogram, cv2.COLOR_BGR2RGB)
            images_unet_og_histogram = cv2.resize(images_histogram_RGB, (1152, 768), interpolation=cv2.INTER_NEAREST)
            pred_mask_og_histogram = test_predict_image_mask_miou(model_unet_2, images_unet_og_histogram, device)

            img_prediction_og_histogram = np.array(pred_mask_og_histogram)

            imgOg_histogram__class_5 = np.where(img_prediction_og_histogram == 1, 255, img_prediction_og_histogram)
            imgOg_histogram__class_5 = cv2.resize(imgOg_histogram__class_5, (w, h), interpolation=cv2.INTER_NEAREST)

            ###################################### end class 5 b3

            images_c1 = images_yolo_c1.copy()
            images_c2 = images_yolo_c2.copy()

            height_joint  = B2_images(predictor_sam, images_c2, imgB_class_2)

            IMG_A, B_distance, BF3_distance, BF4_distance, FINAL_LSQ, FINAL_GRADE_A, FINAL_GRADE_IMAGES_A = analyze_lsq_A(images_og_c1, images_c1, imgA_class_1, imgA_class_2)

            if type_analsze_lsq == 1 :
                IMG_B , _, BF2_distance_cm, FINAL_GRADE_B, FINAL_GRADE_IMAGES_B = analyze_lsq_B_1 (name_file, images_c2, imgB_class_1)
            elif type_analsze_lsq == 2 :
                IMG_B , _, BF2_distance_cm, FINAL_GRADE_B, FINAL_GRADE_IMAGES_B= analyze_lsq_B_2 (images_og_c2, images_c2, imgB_class_2, imgB_class_1, height_joint)
            elif type_analsze_lsq == 3 :
                IMG_B , _, BF2_distance_cm, FINAL_GRADE_B, FINAL_GRADE_IMAGES_B = analyze_lsq_B_3(images_og_c2, images_c2, imgB_class_1, imgOg_histogram__class_5, move_x_loin)
            else :
                IMG_B , _, BF2_distance_cm, FINAL_GRADE_B, FINAL_GRADE_IMAGES_B = analyze_lsq_B_4(images_og_c2, images_c2, imgB_class_1, imgOg_histogram__class_5, move_x_loin, mask_image_c2)


            imag_draw = rectangle_images(images_size, 'A', BF2_distance_cm, FINAL_GRADE_IMAGES_B, B_distance, BF3_distance, BF4_distance, FINAL_LSQ, FINAL_GRADE_IMAGES_A)
            imag_draw = rectangle_images(imag_draw, 'B', BF2_distance_cm, FINAL_GRADE_IMAGES_B, B_distance, BF3_distance, BF4_distance, FINAL_LSQ, FINAL_GRADE_IMAGES_A)
            images_size_og = cv2.resize(imag_draw, (w_og, h_og))
                
            position_A = ( move_y_butt, move_x_butt )
            position_B = ( move_y_loin , move_x_loin )

            images_final = overlay_image(images_size_og, IMG_A, IMG_B, position_A, position_B)

            image_data_base64_og_ = base64.b64encode(cv2.imencode('.jpeg', image_og, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            data_base64_og = (f"data:image/jpeg;base64,{image_data_base64_og_}")
            
            image_data_base64_final_ = base64.b64encode(cv2.imencode('.jpeg', images_final, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            data_base64_Final = (f"data:image/jpeg;base64,{image_data_base64_final_}")

            end_time = time.time()
            elapsed_time = end_time - self.start_time


            json_data = {
                    "timeStamp": name_file, 
                    "bf2": float("{:.2f}".format(BF2_distance_cm)),   
                    "bf2Grading": 'Grade '+str(FINAL_GRADE_B), 
                    "b": float("{:.2f}".format(B_distance)), 
                    "bf3": float("{:.2f}".format(BF3_distance)), 
                    "bf4": float("{:.2f}".format(BF4_distance)), 
                    "lsq": float("{:.2f}".format(FINAL_LSQ)), 
                    "lsqGrading": 'Grade '+str(FINAL_GRADE_A),
                    "imageNoCrop": data_nocrop_base64 ,
                    "originImage": data_base64_og,
                    "image": data_base64_Final,
                    "carcassSide": str(side),
                    "hwId" : str(hwId), 
                    "result" : str(side)+";"+"LSQ "+str(FINAL_GRADE_A)+";"+"BF2 "+str(FINAL_GRADE_B),
                    "lotNo": jobNumber,
                    "workplanId": workplanId,
                    "aiType": aiType
                     }

            if int(save_json) == 1 :
                json_save(name_file, folder_name, json_data)
            
            if int(save_img_result) == 1 :
                cv2.imwrite(f'{folder_name}/images_1/{name_file}.jpg', images_final)

            if int(save_img_all) == 1 :
                cv2.imwrite(f'{folder_name}/images_1/{name_file}.jpg', image_og)
                cv2.imwrite(f'{folder_name}/images_2/{name_file}.jpg', images_final)

            time_stamp_send =  datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            print(json_data)

            try :
                response_final = requests.post(url_output, headers=headers, json=json_data, timeout=1)
                print("time send : "+time_stamp_send+" JSON Response from api", response_final.json()) 
                statusData = "JSON Response from api : True"
            except Exception as e:
                print("time error : "+time_stamp_send+" !!! JSON Response from api : Unable to send data api not work. !!!") 
                statusData = f"JSON Response from api : False, error: {e}"
                pass

        except Exception as e:
            statusData = f"Unable to analyze image, error: {e}"
            pass

        return statusData