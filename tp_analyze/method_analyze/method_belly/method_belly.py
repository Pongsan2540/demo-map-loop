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
type_analsze_belly = config_set_up['parameters-set-up'].getint('type_analsze', None)
folder_name = './'+config_set_up['parameters-set-up'].get('folder_name', None)
save_img_result = 0
save_json = 0
########## config setup end ###########

########## config tempesta start ###########
config_tempesta = configparser.ConfigParser()
config_tempesta.read('./config_tempesta/config_tempesta.conf')
type_ai = config_tempesta['parameters-set-up'].get('aitype', None)
type_ai_list = str(type_ai.lower().split()[0])
########## config setup end ###########

class AnalyzeAPI:

    def __init__(self, data, load_model_all, start_time):
        
        model1, model2, model3, device = load_model_all
        self.model_unet = model1
        self.model_unet_2 = model2
        self.predictor_sam  = model3  
        self.device = device 
        self.json_data = json.loads(data)
        self.start_time = start_time

    def analyze_image(self):

        def morphology_images(images) :
            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(images, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            return closing

        def drawContours_mask(mask_1, mask_2) :
            contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lengthy_array = [len(c) for c in contours]
            position_array = [cv2.boundingRect(c) for c in contours]
            max_index, max_position = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])
            cv2.drawContours(mask_2, contours, max_index, (255,255,255), -1)
            return mask_2
        
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
        
        def round_decimals(x):
            if x - int(x) >= 0.5:
                return int(x) + 1
            else:
                return int(x)  

        def json_save(name_file, folder_name, json_data):

            json_object = json.dumps(json_data, indent=3)
            with open(folder_name+'/'+'output_json'+'/'+str(name_file)+'.json', 'w') as outfile:
                outfile.write(json_object)
            return json_object
        
        ###################################################################################################
        def cal_img_show(image_b):

            size = (300, 1024)
            image_a = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            image_a[:] = (0,0,0)

            max_width_a = image_a.shape[1]
            max_height_a = image_a.shape[0] 

            if image_b.shape[0] < max_height_a and image_b.shape[1] < max_width_a:
                text_cal_img = "not resize"
                resized_image_b = image_b
            else:
                aspect_ratio_b = image_b.shape[1] / image_b.shape[0]
                if aspect_ratio_b > (max_width_a / max_height_a):
                    new_width_b = max_width_a
                    new_height_b = int(max_width_a / aspect_ratio_b)
                else:
                    new_height_b = max_height_a
                    new_width_b = int(max_height_a * aspect_ratio_b)
                resized_image_b = cv2.resize(image_b, (new_width_b, new_height_b))
                text_cal_img = f"resize : {aspect_ratio_b:.2f}"

            canvas = image_a

            y_offset = int((max_height_a - resized_image_b.shape[0]) // 2)
            x_offset = int((max_width_a - resized_image_b.shape[1]) // 2)

            canvas[y_offset:y_offset + resized_image_b.shape[0], x_offset:x_offset + resized_image_b.shape[1]] = resized_image_b
            cv2.putText(canvas, text_cal_img, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

            return canvas
        ###################################################################################################

                                    
        url_output = self.json_data.get("URL_SEND", None)
        time_stamp = self.json_data.get("TIME_STAMP", None)
        name_file = self.json_data.get("NAME_FILE", None)
        jobNumber = self.json_data.get("JOB_NUMBER", None)
        hwId = self.json_data.get("HW_ID", None)
        workplanId = self.json_data.get("WORKPLAN_ID", None)
        data_detail = self.json_data.get("DATA_DERAIL", None)
        aiType = self.json_data.get("AI_TYPE", None)

        if url_output is None:
            url_output = "http://"+by_pass_url+":"+by_pass_port+"/"+by_pass_name
            aiType = type_ai_list
        else:
            pass

        if isinstance(data_detail, list) and data_detail:
            best_name_label, x1, y1, x2, y2, best_prob, _ = data_detail
            pass
        else:
            data_detail = None  
        images_value = self.json_data.get("IMAGE", None)

        try:
            pattern = rb'data:image/jpeg;base64,'
            images_value = re.sub(pattern, b'', images_value.encode())
            image_array = np.frombuffer(base64.b64decode(images_value), dtype=np.uint8)
            image_nocrop = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            image_crop_from_api =  image_nocrop[y1:y2, x1:x2]          

            image_nocrop_data_base64 = base64.b64encode(cv2.imencode('.jpeg', image_nocrop,
                                            [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            data_nocrop_base64 = (f"data:image/jpeg;base64,{image_nocrop_data_base64}")

            image = image_crop_from_api.copy()
            image_og = image.copy()
            
            #image = image
            #image_data_base64_2 = base64.b64encode(cv2.imencode('.jpeg', image_og, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            #data_base64_2 = (f"data:image/jpeg;base64,{image_data_base64_2}")

            h, w,_ = image.shape

            if type_analsze_belly == 1 :

                ###### SAM BG S #################
                Image_pred = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Image_pred = cv2.resize(Image_pred, (1152, 768), interpolation=cv2.INTER_NEAREST)

                input_box = np.array([x1, y1, x2, y2])
                self.predictor_sam.set_image(image_nocrop)
                masks, scores, _ = self.predictor_sam.predict(
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
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                lengthy_array = [len(c) for c in contours]
                position_array = [cv2.boundingRect(c) for c in contours]
                max_index, max_position = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])
                        
                closing = np.zeros_like(closing)
                cv2.drawContours(closing, contours, max_index, (255,255,255), -1)

                test_closing = closing.copy()
                test_closing = cv2.cvtColor(test_closing,cv2.COLOR_GRAY2RGB)
                cv2.rectangle(test_closing, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_4)

                make_img =  closing[y1:y2, x1:x2]    
                ###### SAM BG E #################
            else :

                ###### UNET BG S #################

                Image_pred = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Image_pred = cv2.resize(Image_pred, (1152, 768), interpolation=cv2.INTER_NEAREST)
            
                pred_mask_bg = test_predict_image_mask_miou(self.model_unet_2, Image_pred, self.device)
                img_prediction = np.array(pred_mask_bg)

                img_class_bg = np.where(img_prediction == 1, 255, img_prediction)
                img_class_bg = np.where(img_class_bg != 255, 0, img_class_bg)
                
                img_class_bg = cv2.resize(img_class_bg, (w, h), interpolation=cv2.INTER_NEAREST)    
                img_class_bg_binary = img_class_bg.astype(np.uint8)

                closing = morphology_images(img_class_bg_binary)

                make_img = np.zeros((h ,w),dtype=np.uint8)
                make_img = drawContours_mask(closing, make_img)
                ###### UNET BG E #################

            pred_mask = test_predict_image_mask_miou(self.model_unet, Image_pred, self.device)

            img_prediction = np.array(pred_mask)

            img_class_1 = np.where(img_prediction == 1, 255, img_prediction)
            img_class_2 = np.where(img_prediction == 2, 255, img_prediction)

            img_class_1 = np.where(img_class_1 == 2, 0, img_class_1)
            img_class_2 = np.where(img_class_2 == 1, 0, img_class_2)

            img_class_1 = cv2.resize(img_class_1, (w, h), interpolation=cv2.INTER_NEAREST)
            img_class_2 = cv2.resize(img_class_2, (w, h), interpolation=cv2.INTER_NEAREST)

            img_class_1_binary = img_class_1.astype(np.uint8)
            img_class_2_binary = img_class_2.astype(np.uint8)

            contours_img_class_1, hierarchy = cv2.findContours(img_class_1_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            closing_binary = morphology_images(img_class_2_binary)
            contours_img_class_2, hierarchy = cv2.findContours(closing_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours_img_class_2):
                cv2.drawContours(make_img, [contour], 0, (0,0,0), -1)

            make_img_5 = np.zeros((h ,w),dtype=np.uint8)
            make_img_6 = drawContours_mask(make_img, make_img_5)
            make_img = make_img_6

            count_255_bg = np.count_nonzero(make_img_5 == 0)

            make_img_2 = make_img.copy()
            for i, contour in enumerate(contours_img_class_1):
                    cv2.drawContours(make_img_2, [contour], 0, (0,0,0), -1)

            img_class_1_binary = cv2.bitwise_and(img_class_1_binary,make_img)

            count_255_class1 = np.count_nonzero(img_class_1_binary == 255)
            count_255_class2 = np.count_nonzero(make_img_2 == 255)
            count_255_all = count_255_class1+count_255_class2

            per_sead = (20*count_255_bg)/100
            if count_255_class2 < per_sead :
                checkImages  = False
            else :
                checkImages  = True

            #make_img_final_morphology2 = morphology_images(make_img_2)

            persenMeat = (count_255_class1*100)/count_255_all
            persenMeat = round_decimals(persenMeat)
            persenfat = abs(100-persenMeat)
                
            if persenMeat >= 60:
                grade = 'A+'
            elif persenMeat >= 50:
                grade = 'A'
            elif persenMeat >= 40:
                grade = 'B'
            elif persenMeat >= 30:
                grade = 'C'
            else:
                grade = 'D'

            p_meat_text = int(persenMeat)
            p_fat_text = int(persenfat)

            #p_meat_text = float("{:.0f}".format(persenMeat))
            #p_fat_text = float("{:.0f}".format(persenfat))
            
            img_final_class_1 = cv2.bitwise_and(image_og, image_og, mask=img_class_1_binary)
            img_final = cv2.bitwise_and(image_og, image_og, mask=make_img)

            image_data_base64_2 = base64.b64encode(cv2.imencode('.jpeg', img_final, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            data_base64_2 = (f"data:image/jpeg;base64,{image_data_base64_2}")

            img_final = cv2.addWeighted(img_final_class_1, 0.5, img_final, 0.5, 10)

            text_str = f"meat = {p_meat_text}%   fat = {p_fat_text}%  grade {grade}, date : {str(time_stamp)}"
            black_image = np.zeros((35, 1000, 3), dtype=np.uint8)
            x, y, w_draw, h_draw = 0, 5, 600, 25
            cv2.rectangle(black_image, (x, y), (x + w_draw, y + h_draw), (218, 218, 218), -1)    
            cv2.putText(black_image, text_str, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            black_image_resized = cv2.resize(black_image, (w, 35))

            if len(img_final.shape) == 2: 
                img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2BGR)

            images_final = cv2.vconcat([img_final, black_image_resized])

            #image_data_base64 = base64.b64encode(cv2.imencode('.jpeg', images_final,[cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            #data_base64 = (f"data:image/jpeg;base64,{image_data_base64}")

            ###############
            try :
                img_show = cal_img_show(images_final)
                #cv2.imwrite("./output/images_10/"+name_file+".jpg", img_show)
            except Exception as e:
                statusData = f"Error cal image : {e}"
                img_show = images_final
                pass

            image_data_base64 = base64.b64encode(cv2.imencode('.jpeg', img_show,
                                            [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
            data_base64 = (f"data:image/jpeg;base64,{image_data_base64}")

            #################


            end_time = time.time()
            elapsed_time = end_time - self.start_time

            json_data = {
                "timeStamp": str(time_stamp), 
                "meat": int(persenMeat),
                "fat": int(persenfat),
                "values": (f'{p_meat_text}/{p_fat_text}'),
                "result": "Grade " + grade, 
                "imageNoCrop": data_nocrop_base64 ,
                "image": data_base64 ,
                "originImage": data_base64_2, 
                "processTime": round(elapsed_time,4),
                "hwId": str(hwId),
                "checkImages": checkImages,
                "lotNo": jobNumber,
                "workplanId": workplanId,
                "aiType": aiType
                }
            
            if int(save_json) == 1 :
                json_save(name_file, folder_name, json_data)
            
            if int(save_img_result) == 1 :
                cv2.imwrite(f'{folder_name}/images_2/{name_file}.jpg', images_final)

            if int(save_img_all) == 1 :
                cv2.imwrite(f'{folder_name}/images_1/{name_file}.jpg', image_og)
                cv2.imwrite(f'{folder_name}/images_2/{name_file}.jpg', img_final)
                cv2.imwrite(f'{folder_name}/images_3/{name_file}.jpg', images_final)

            time_stamp_send =  datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

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





