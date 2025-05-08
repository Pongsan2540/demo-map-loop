################################################################################
# script python of API warmApi version date 20240108
#               - add analyze_lsq_B_3
################################################################################

import numpy as np
import cv2
import statistics
import math
from math import radians, sin, cos, sqrt, acos, degrees
import math
from PIL import Image
import traceback

def move_points(x1, y1, x2, y2, new_x1, new_y1):
    # Calculate the distance and angle between the two points
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = math.atan2(y2 - y1, x2 - x1)
    # Calculate the new coordinates based on the relative distance and orientation
    new_x2 = int(new_x1 + distance * math.cos(angle))
    new_y2 = int(new_y1 + distance * math.sin(angle))
    return new_x2, new_y2

def rectangle_images(images_size, type_carcass, BF2_distance_cm, FINAL_GRADE_B, B_distance, BF3_distance, BF4_distance, FINAL_LSQ, FINAL_GRADE_A):    
    if type_carcass == 'A' :
        P_draw = [(5,5,500,330), (5,350,500,80)]
        put_Text = ['BF3 : '+str(BF3_distance), 'BF4 : '+str(BF4_distance), 'B : '+str(B_distance), 
                        'LSQ : '+str(FINAL_LSQ), 'Grade of LSQ : '+str(FINAL_GRADE_A)]
        size_text = [(20, 80), (20, 160), (20, 240), (20, 320), (20, 410)]
        color_text  = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 255, 255)]
    elif type_carcass == 'B' :
        P_draw = [(940,5,470,100), (940,120,470,80)]
        put_Text = ['BF2 : '+str(BF2_distance_cm), 'Grade of BF2 '+str(FINAL_GRADE_B)]
        size_text = [(955, 80), (955, 170)]
        color_text  = [(0, 0, 0), (255, 255, 255)]
    color_draw = [(240,240,240), (0,0,192)]
    for i in range(len(P_draw)) :
        x,y,w,h = P_draw[i]
        cv2.rectangle(images_size, (x, y), (x + w, y + h),color_draw[i] , -1)   
    for i_text in range(len(put_Text)):
        cv2.putText(images_size, put_Text[i_text], size_text[i_text], cv2.FONT_HERSHEY_SIMPLEX,1.5, color_text[i_text], 5, cv2.LINE_AA)    
    return images_size

def overlay_image(images_size_og, original_A, original_B, position_A, position_B):
    if position_A != False and position_B != False :
        h_A, w_A, _ = original_A.shape
        h_B, w_B, _ = original_B.shape
        images_size_B = Image.fromarray(images_size_og)
        images_size_B.paste(Image.fromarray(original_B), position_B)
        images_size_A = images_size_B.copy()
        images_size_A.paste(Image.fromarray(original_A), position_A)
        result_np_final = np.array(images_size_A)

        if position_A[1]+h_A > position_B[1] :
            h_A = abs(h_A-abs((position_A[1]+h_A) - (position_B[1])))

        cv2.rectangle(result_np_final, pt1=position_B, pt2=(position_B[0]+w_B,position_B[1]+h_B), color=(0,0,255), thickness=3)
        cv2.rectangle(result_np_final, pt1=position_A, pt2=(position_A[0]+w_A,position_A[1]+h_A), color=(0,0,255), thickness=3)
    elif position_A != False and position_B == False :
        h_A, w_A, _ = original_A.shape
        images_size_og = Image.fromarray(images_size_og)
        original_A_pil = Image.fromarray(original_A)
        images_size_og.paste(original_A_pil, position_A)
        result_np_final = np.array(images_size_og)
        cv2.rectangle(result_np_final, pt1=position_A, pt2=(position_A[0]+w_A,position_A[1]+h_A), color=(0,0,255), thickness=2)
    elif position_A == False and position_B != False :
        h_B, w_B, _ = original_B.shape
        images_size_og = Image.fromarray(images_size_og)
        original_B_pil = Image.fromarray(original_B)
        images_size_og.paste(original_B_pil, position_B)
        result_np_final = np.array(images_size_og)
        cv2.rectangle(result_np_final, pt1=position_B, pt2=(position_B[0]+w_B,position_B[1]+h_B), color=(0,0,255), thickness=2)
    else  :
        result_np_final = images_size_og
    return result_np_final

def find_angle(x1, y1, x2, y2, x3, y3):
    
    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)

    ab = math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    ac = math.sqrt((A[0] - C[0])**2 + (A[1] - C[1])**2)
    bc = math.sqrt((B[0] - C[0])**2 + (B[1] - C[1])**2)

    cos_a = (ab**2 + ac**2 - bc**2) / (2 * ab * ac)
    sin_a = math.sqrt(1 - cos_a**2)

    return math.acos(cos_a), math.asin(sin_a)

def images_equalizeHist(images):
    B, G, R = images[:, :, 0], images[:, :, 1], images[:, :, 2]  
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    output_images = np.dstack((B, G, R))
    return output_images

def angle_between_three_points(p1, p2, p3):
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    angle = angle2 - angle1
    return math.degrees(angle)

def rotate_point_wrt_center(point_to_be_rotated, angle, center_point = (0,0)):        
    angle = radians(angle)      
    xnew = int(cos(angle)*(point_to_be_rotated[0] - center_point[0]) - 
                   sin(angle)*(point_to_be_rotated[1] - center_point[1]) + center_point[0])
    ynew = int(sin(angle)*(point_to_be_rotated[0] - center_point[0]) + 
                   cos(angle)*(point_to_be_rotated[1] - center_point[1]) + center_point[1] )  
    return (xnew, ynew)

def convert_rgb(image):
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image
    return out

def select_objects(image):
    thresh, blackAndWhiteImage = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinate_position = [cv2.boundingRect(cnt) for cnt in contours]
    return (coordinate_position,contours)

def analyze_lsq_A(Image_og, originalImage, mask_image_A_1, mask_image_A_2):     

    og_images = Image_og
    # Input data image (original, mask cube and mask spinal)
    mask_image_A_1 = mask_image_A_1.astype(np.uint8)
    mask_image_A_2 = mask_image_A_2.astype(np.uint8)

    try:
        kernel = np.ones((5,5),np.uint8)
        # Images A1
        opening = cv2.morphologyEx(mask_image_A_1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        blur_image = cv2.blur (closing, ksize= (1,1)) #
        thresh, blackAndWhiteImage = cv2.threshold(blur_image, 10, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lengthy_array = [len(c) for c in contours]
        position_array = [cv2.boundingRect(c) for c in contours]
        max_index_A1, max_position_A1 = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])

        max_position_center_A1 = int(max_position_A1[3]/2)

        mask_image_A_1 = np.zeros_like(mask_image_A_1)
        cv2.drawContours(mask_image_A_1, contours, max_index_A1, (255,255,255), -1)

        # Images A2
        opening = cv2.morphologyEx(mask_image_A_2, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        blur_image = cv2.blur (closing, ksize= (1,1)) #
        thresh, blackAndWhiteImage = cv2.threshold(blur_image, 10, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lengthy_array = [len(c) for c in contours]
        position_array = [cv2.boundingRect(c) for c in contours]
        max_index, max_position = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])
        max_index_A2, max_position_A2 = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])

        mask_image_A_2 = np.zeros_like(mask_image_A_2)
        cv2.drawContours(mask_image_A_2, contours, max_index, (255,255,255), -1)

        high, width = mask_image_A_2.shape
        left_distance = abs(max_position[0])
        right_distance = abs((max_position[0] + max_position[2]) - width)  

        if max_position_A2[1] < max_position_center_A1  \
           and (max_position_A2[1]+max_position_A2[3]) > (max_position_A1[1]+max_position_A1[3]) :

            if left_distance > right_distance :
                Image_og = cv2.flip(Image_og, 1)
                originalImage = cv2.flip(originalImage, 1)
                mask_image_A_1 = cv2.flip(mask_image_A_1, 1)
                mask_image_A_2 = cv2.flip(mask_image_A_2, 1)

                opening = cv2.morphologyEx(mask_image_A_1, cv2.MORPH_OPEN, kernel)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                thresh, blackAndWhiteImage = cv2.threshold(closing, 10, 255, cv2.THRESH_BINARY)

                contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                lengthy_array = [len(c) for c in contours]
                position_array = [cv2.boundingRect(c) for c in contours]
                max_index, max_position = max(enumerate(position_array), key=lambda x: lengthy_array[x[0]])

                mask_image_A_1 = np.zeros_like(mask_image_A_1)
                cv2.drawContours(mask_image_A_1, contours, max_index, (255,255,255), -1)

                IMAGES_DIRECTION = "L"
            else :
                IMAGES_DIRECTION = "R"

            gray_image = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
            blur_image = cv2.blur (gray_image, ksize= (10,10)) #
            thresh, blackAndWhiteImage = cv2.threshold(blur_image, 10, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            erosion = cv2.erode(closing,kernel,iterations = 1)
            cont = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            edge_image = np.zeros(originalImage.shape, dtype=np.uint8)
            edge_image = cv2.drawContours(edge_image, cont[0], -1, (255, 255, 255),1)

            # Overlay images
            overlay_image = cv2.addWeighted(mask_image_A_1, 1, mask_image_A_2, 1, 0)
            height, width, channels = originalImage.shape

            image_resize_pork_loaf = cv2.resize(mask_image_A_1, (width, height))
            image_resize_pork_loin = cv2.resize(mask_image_A_2, (width, height))
            img_resize = cv2.resize(overlay_image, (width, height))

            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(img_resize, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            thresh, blackAndWhiteImage = cv2.threshold(closing, 10, 255, cv2.THRESH_BINARY)

            mask_image_analyze = blackAndWhiteImage
            mask_image_analyze = cv2.GaussianBlur(mask_image_analyze, (65,65), 0)
            _, mask_image_analyze = cv2.threshold(mask_image_analyze, 130, 255, 0)

            contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            coordinate_position =[]
            hight_position =[]
            for i in range(len(contours)):
                x,y,w,h = cv2.boundingRect(contours[i])
                coordinate_position.append([x, y, x+w, y+h])
                hight_position.append(h)

            pork_loaf_position = coordinate_position.pop(hight_position.index(min(hight_position)))
            crop_pork_loaf = mask_image_analyze[pork_loaf_position[1]:pork_loaf_position[3], pork_loaf_position[0]:pork_loaf_position[2]]

            white_pixels = np.array(np.where(crop_pork_loaf == 255))
            first_white_pixel = white_pixels[:,0]
            last_white_pixel = white_pixels[:,-1]    

            x_upper, y_upper = (pork_loaf_position[0] + first_white_pixel[1]),(pork_loaf_position[1] + first_white_pixel[0])
            x_lower, y_lower = (pork_loaf_position[0] + last_white_pixel[1]),(pork_loaf_position[1] + last_white_pixel[0])
            x_center, y_center = int((x_lower + x_upper)/2), int((y_lower +y_upper)/2)
             # calculate distance condition
            distance_condition = pork_loaf_position[2] - pork_loaf_position[0]
            # calculate distance to center
            distance_center = mask_image_analyze[y_center, :(x_center - distance_condition)]
            i_center = len(distance_center) - 1 - next((i for i, val in enumerate(reversed(distance_center)) if val == 255), -1)
            # calculate distance to lower
            distance_lower = mask_image_analyze[y_lower, :(x_lower - distance_condition)]
            i_lower = len(distance_lower) - 1 - next((i for i, val in enumerate(reversed(distance_lower)) if val == 255), -1)

            p1, p2, p3 = (i_center, y_center), (i_lower, y_lower), (x_lower, y_lower)
            angle = 90 - (angle_between_three_points(p1, p2, p3))
            # Define the starting point and angle in degrees
            start_point = (x_lower, y_lower)
            # Convert the angle to radians
            angle_radians = math.radians(angle)
            # Define the length of the line segment
            length = abs(i_lower-x_lower)
            # Calculate the endpoint of the line segment
            end_point = (start_point[0] - length * math.cos(angle_radians),
                                 start_point[1] - length * math.sin(angle_radians))   

            # Define the two lines as pairs of points
            line1 = ((int(end_point[0]),int(end_point[1])), (x_lower, y_lower ))
            line2 = ((i_lower, y_lower), (i_center, y_center))

            # Calculate the slopes and intercepts of the two lines
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            if abs(x2 - x1) == 0 :
                slope1 = 1
                intercept1 = y1 - slope1 * x1
            else :
                slope1 = (y2 - y1) / (x2 - x1)
                intercept1 = y1 - slope1 * x1

            x1, y1 = line2[0]
            x2, y2 = line2[1]
            if abs(x2 - x1) == 0 :
                slope2 = 1
                intercept2 = y1 - slope2 * x1
            else :
                slope2 = (y2 - y1) / (x2 - x1)
                intercept2 = y1 - slope2 * x1

            x = abs((intercept2 - intercept1)) / abs((slope1 - slope2))
            y = slope1 * x + intercept1
            intersection = (int(x), int(y))

            # Draw lines
            cv2.line(Image_og, (intersection), (x_lower, y_lower), (0, 0, 192), thickness=5)
            # Draw circles
            cv2.circle(Image_og,(x_lower,y_lower), 1, (0, 0, 192), 15)
            cv2.circle(Image_og,(intersection), 1, (0, 0, 192), 15)   

            # Parts of BF3, BF4 and BF5
            p1, p2, p3 = (x_upper, y_upper), (x_lower, y_lower), (i_lower, y_lower)   
            angle_2 = abs(angle_between_three_points(p1, p2, p3))

            mask_image_A_1_resize = cv2.resize(mask_image_A_1, (width, height))
            thresh, blackAndWhiteImage = cv2.threshold(mask_image_A_1_resize, 10, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            mask_image_A_1_bw = closing

            thresh, blackAndWhiteImage_edge = cv2.threshold(edge_image, 10, 255, cv2.THRESH_BINARY)
            edge_image_bw = blackAndWhiteImage_edge

            h, w = mask_image_A_1_bw.shape
            center_point = (int(w/2), int(h/2))

            M = cv2.getRotationMatrix2D(center_point,(abs(360-angle_2)-90),1)

            rotated_originalImage = cv2.warpAffine(originalImage,M,(w,h)) 
            rotated_image_resize_pork_loaf = cv2.warpAffine(mask_image_A_1_bw,M,(w,h)) 
            rotated_edge_image = cv2.warpAffine(edge_image_bw,M,(w,h)) 

            position_lower_refer = rotate_point_wrt_center((x_lower, y_lower),-(abs(360-angle_2)-90),center_point = center_point)
            x_lower_refer, y_lower_refer = position_lower_refer

            cv2.circle(rotated_originalImage, (position_lower_refer[0],position_lower_refer[1]), 1, (0,0,192), 15)

            gray_rotated_edge_image = cv2.cvtColor(rotated_edge_image,cv2.COLOR_BGR2GRAY)
            thresh, blackAndWhiteImage_rotated_edge_image = cv2.threshold(gray_rotated_edge_image, 90, 255, cv2.THRESH_BINARY)     
            closing_edge_image = cv2.morphologyEx(blackAndWhiteImage_rotated_edge_image, cv2.MORPH_CLOSE, kernel)

            j_refer_bf3 = np.argmax(closing_edge_image[y_lower_refer, x_lower_refer:]) + x_lower_refer

            thresh, blackAndWhiteImage_rotated_image_resize_pork_loaf = cv2.threshold(rotated_image_resize_pork_loaf, 100, 255, cv2.THRESH_BINARY)     
            erosion_image_resize = cv2.erode(blackAndWhiteImage_rotated_image_resize_pork_loaf,kernel,iterations = 1)

            contours, hierarchy = cv2.findContours(erosion_image_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            coordinate_position = [cv2.boundingRect(cnt) for cnt in contours]

            max_hight_index = max(enumerate(coordinate_position), key=lambda x: x[1][-1])[0]
            position_value = coordinate_position[max_hight_index]

            position_white_x2 = int((position_value[0]+position_value[2])-1)

            erosion_image_resize[:,position_white_x2]
            data_white = [i for i in range(len(erosion_image_resize[:,position_white_x2])) if erosion_image_resize[:,position_white_x2][i] == 255]
            position_white_y2 = int(statistics.median(data_white))

            j_refer_bf4 = np.argmax(closing_edge_image[position_white_y2, position_white_x2:]) + position_white_x2

            bf4_upper_point = (position_white_x2, position_white_y2)
            bf4_edge_point = (j_refer_bf4-3, position_white_y2)

            bf3_lower_point = (position_lower_refer[0], position_lower_refer[1])
            bf3_edge_point =  (j_refer_bf3-3, y_lower_refer)

            cv2.circle(rotated_originalImage, bf3_edge_point, 1, (0, 0, 192), 15)
            cv2.circle(rotated_originalImage, bf4_upper_point, 1, (0, 0, 192), 15)
            cv2.circle(rotated_originalImage, bf4_edge_point, 1, (0, 0, 192), 15)

            bf3_lower_point_final = rotate_point_wrt_center(bf3_lower_point, abs(360-angle_2)-90,center_point = center_point)
            bf3_edge_point_final = rotate_point_wrt_center(bf3_edge_point, abs(360-angle_2)-90,center_point = center_point)
            bf4_upper_point_final = rotate_point_wrt_center(bf4_upper_point, abs(360-angle_2)-90,center_point = center_point)
            bf4_edge_point_final = rotate_point_wrt_center(bf4_edge_point, abs(360-angle_2)-90,center_point = center_point)

            cv2.circle(Image_og, bf3_lower_point_final, 1, (0, 0, 192), 15)
            cv2.circle(Image_og, bf3_edge_point_final, 1, (0, 0, 192), 15)
            cv2.circle(Image_og, bf4_upper_point_final, 1, (0, 0, 192), 15)
            cv2.circle(Image_og, bf4_edge_point_final, 1, (0, 0, 192), 15)

            cv2.line(Image_og, bf3_lower_point_final, bf3_edge_point_final, (0, 0, 192), thickness=5)
            cv2.line(Image_og, bf4_upper_point_final, bf4_edge_point_final, (0, 0, 192), thickness=5)

            BF3_X12 = bf3_lower_point_final[0] - bf3_edge_point_final[0]
            BF3_Y12 = bf3_lower_point_final[1] - bf3_edge_point_final[1]
            BF4_X12 = bf4_upper_point_final[0] - bf4_edge_point_final[0]
            BF4_Y12 = bf4_upper_point_final[1] - bf4_edge_point_final[1]

            B_distance = float("{:.2f}".format(math.sqrt((intersection[0] - x_lower)**2 + (intersection[1] - y_lower)**2)))
            BF3_distance = float("{:.2f}".format(math.sqrt((BF3_X12)**2 + (BF3_Y12)**2)))
            BF4_distance = float("{:.2f}".format(math.sqrt((BF4_X12)**2 + (BF4_Y12)**2)))
            if B_distance != 0:
                FINAL_LSQ = float("{:.2f}".format((BF3_distance + BF4_distance) / (2 * B_distance)))
            else:
                FINAL_LSQ = 0  # or any other appropriate value    

            FINAL_GRADE = next((grade for boundary, grade in [
                (0.20, "A"+":<0.20"),
                (0.26, "B"+":0.21-0.26"),
                (0.32, "C"+":0.27-0.32"),
                (0.38, "D"+":0.33-0.38"),
                (0.44, "E"+":0.39-0.44")
            ]if FINAL_LSQ <= boundary), "F"+":>0.45")
       
            FINAL_GRADE_IMAGES = next((grade for boundary, grade in [(0.20, "A"), (0.26, "B"), (0.32, "C"), (0.38, "D"), (0.44, "E")] \
                                                            if FINAL_LSQ <= boundary ), "F")   

            if IMAGES_DIRECTION == 'L' :
                    originalImage = cv2.flip(originalImage, 1)     
                    Image_og = cv2.flip(Image_og, 1)     

        else:
            B_distance, BF3_distance, BF4_distance, FINAL_LSQ = [0] * 4
            FINAL_GRADE = "N/A"  
            FINAL_GRADE_IMAGES = "N/A" 
            Image_og = og_images
    except:
        B_distance, BF3_distance, BF4_distance, FINAL_LSQ = [0] * 4
        FINAL_GRADE = "N/A"  
        FINAL_GRADE_IMAGES = 'N/A'  
        Image_og = og_images

    return (Image_og, B_distance, BF3_distance, BF4_distance, FINAL_LSQ, FINAL_GRADE, FINAL_GRADE_IMAGES)   

def analyze_lsq_B_1 (Image_og, original_images, mask_images_loin) :

    og_images = original_images
    mask_images_loin = mask_images_loin.astype(np.uint8)

    try:
        h, w, channels = original_images.shape
        image_resize = cv2.resize(mask_images_loin, (w, h))        
            
        aspect_ratio = 16
        half = h//aspect_ratio
        
        variable_aspect = [f"aspect_ratio_{i+1}" for i in range(aspect_ratio)]      
        data_position = [[position, position+half] for position in range(0, len(variable_aspect)*half, half)]
        
        final_position = sorted(list(set(data_position[5] + data_position[6])))
        final_position.pop(len(final_position) // 2)
        
        length = final_position[1]-final_position[0]
        crop_mask = image_resize[final_position[0] : final_position[0]+length, 0 : w]
        crop_original =  original_images[final_position[0] : final_position[0]+length, 0 : w]
        
        h_crop, w_crop = crop_mask.shape
        
        thresh, blackAndWhiteImage = cv2.threshold(crop_mask, 100, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        erosion = cv2.erode(closing,kernel,iterations = 1)
        
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        coordinate_position = [cv2.boundingRect(cnt) for cnt in contours]

        if not coordinate_position:

            print("coordinate_position is empty")
            BF2_distance_pixel, BF2_distance_cm, FINAL_GRADE = [0] * 3     

        else :

            max_hight_index = max(enumerate(coordinate_position), key=lambda x: x[1][-1])[0]
            position_value = coordinate_position[max_hight_index]
            
            x_object = position_value[0]
            x_start = 0
            x_end = w

            left_spacing = abs(x_object - x_start)
            right_spacing = abs(x_end - x_object)
            
            IMAGES_DIRECTION = "right_side" if left_spacing > right_spacing else "left_side"
            # Create a new image equal to the original image.
            mask_image_create = np.zeros_like(crop_mask)
            # Specify the coordinates of the image of interest.
            cv2.drawContours(mask_image_create, contours, max_hight_index, (255,255,255), -1)
            
            # System check that the picture is missing a lot or not
            lower_bound = int((h_crop*10)/100)
            upper_bound = int((h_crop*90)/100)
                
            if position_value[1] < lower_bound and position_value[3] > upper_bound :
                data_num_white = np.count_nonzero(mask_image_create == 255, axis=1)
                median = int(np.median(np.where(data_num_white == np.min(data_num_white))[0]))
                BF2_distance_pixel =  list(mask_image_create[median,:]).count(255)
                INDEX_white_first =  list(mask_image_create[median,:]).index(255)
                INDEX_white_last = INDEX_white_first + BF2_distance_pixel
                
                calibration_1_pixel_per_1_cm = 0.03577464788732394
                BF2_distance_cm = int((BF2_distance_pixel * calibration_1_pixel_per_1_cm)*10)
                    
                FINAL_GRADE = "A+"+":<20" if BF2_distance_cm <= 20 \
                                        else "A"+":21-24" if BF2_distance_cm <= 24 \
                                        else "B"+":25-28" if BF2_distance_cm <= 28 \
                                        else "C"+":>29" if BF2_distance_cm >= 29 \
                                        else "N/A"
                
                FINAL_GRADE_IMAGES = "A+" if BF2_distance_cm <= 20 \
                                        else "A" if BF2_distance_cm <= 24 \
                                        else "B" if BF2_distance_cm <= 28 \
                                        else "C" if BF2_distance_cm >= 29 \
                                        else "N/A"
                
                cv2.line(original_images, (INDEX_white_first, final_position[0]+median), 
                        (INDEX_white_last,final_position[0]+median), (0, 0, 192), thickness=2) #2
                        
                cv2.circle(original_images,  (INDEX_white_first, final_position[0]+median), 1, (0,0,192), 5)
                cv2.circle(original_images,  (INDEX_white_last,final_position[0]+median), 1, (0,0,192), 5)

            else :
                BF2_distance_pixel, BF2_distance_cm = [0] * 2  
                FINAL_GRADE = "N/A"
                FINAL_GRADE_IMAGES = "N/A"
                original_images = og_images

    except:
        BF2_distance_pixel, BF2_distance_cm = [0] * 2
        FINAL_GRADE = "N/A"
        FINAL_GRADE_IMAGES = "N/A"
        original_images = og_images

    return (original_images ,BF2_distance_pixel,BF2_distance_cm, FINAL_GRADE, FINAL_GRADE_IMAGES)

def analyze_lsq_B_2(Image_og, originalImage, mask_image_B_1, mask_image_B_2, hight_max):  

    og_images = Image_og
    maskImage = mask_image_B_1.astype(np.uint8)
    maskImage_2 = mask_image_B_2.astype(np.uint8)
    
    try:
        gray_og = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
        high, width = gray_og.shape
        image_resize_maskImage = cv2.resize(maskImage, (width, high))
        image_resize_maskImage_2 = cv2.resize(maskImage_2, (width, high))

        # Select the largest group of section C 
        coordinate_position, contours = select_objects(image_resize_maskImage)
        max_hight_index = max(enumerate(coordinate_position), key=lambda x: x[1][-1])[0]
        position_value = coordinate_position[max_hight_index]
        mask_image_create = np.zeros_like(originalImage)
        cv2.drawContours(mask_image_create, contours, max_hight_index, (255,255,255), -1)

        # Select the largest group of section CB
        coordinate_position, contours = select_objects(image_resize_maskImage_2)
        max_hight_index = max(enumerate(coordinate_position), key=lambda x: x[1][-1])[0]
        position_value = coordinate_position[max_hight_index]
        mask_image_create_B = np.zeros_like(originalImage)
        cv2.drawContours(mask_image_create_B, contours, max_hight_index, (255,255,255), -1)

        left_img = 0   
        right_img = width
        x1 = position_value[0]
        x2 = x1+position_value[2]

        difference_left = abs(x1 - left_img)
        difference_right = abs(right_img - x2) 

        if difference_left > difference_right :
            mask_image_c = mask_image_create
            mask_image_b = mask_image_create_B
            IMAGES_DIRECTION = "right_side"
        else :         
            Image_og = cv2.flip(Image_og, 1)   
            originalImage = cv2.flip(originalImage, 1)
            mask_image_c = cv2.flip(mask_image_create, 1)
            mask_image_b = cv2.flip(mask_image_create_B, 1)
            IMAGES_DIRECTION = "left_side"

        gray_originalImage = cv2.cvtColor(mask_image_c,cv2.COLOR_BGR2GRAY)
        thresh, blackAndWhiteImage = cv2.threshold(gray_originalImage, 100, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        coordinate_position = [cv2.boundingRect(cnt) for cnt in contours]
        first_white = coordinate_position[0][0]

        data_white = []
        for i in range(len(blackAndWhiteImage[:,first_white])) :
            a =  blackAndWhiteImage[i,first_white]
            if blackAndWhiteImage[i,first_white] == 255:
                data_white.append(i)
        #position_median = int(statistics.median(data_white)) #use center point
        position_median = min(data_white) #use min point
        white_turning_point = (first_white, position_median)

        h, w = blackAndWhiteImage.shape
        condition_upper = int((5*h)/100)
        list_condition_upper = list(blackAndWhiteImage[condition_upper,:])

        if sum(list_condition_upper) > 0 :
            x1_upper = list_condition_upper.index(255)
            
            first_position = (x1_upper,condition_upper)
            arc_position = white_turning_point
            reference_position= (int(width/2), position_median)

            p1, p2, p3 = first_position, arc_position, reference_position
            angle = abs(90 - angle_between_three_points(p1, p2, p3))

            data_white = [np.sum(blackAndWhiteImage[i,:] == 255) for i in range(blackAndWhiteImage.shape[0])]
            max_num_white = max(data_white)

            test_Max = (first_white, abs(position_median-(max_num_white)*11))
            a_Max = rotate_point_wrt_center(test_Max, angle, center_point = white_turning_point)    

            #use sam model
            #hight_max = 100
            up_roi = abs(a_Max[1])-hight_max
            down_roi = abs(a_Max[1])+hight_max

            gray_mask_b = cv2.cvtColor(mask_image_b,cv2.COLOR_BGR2GRAY)

            data_num_white = []
            for i in range(up_roi, down_roi):
                row_white = gray_mask_b[i,:]
                pixels = np.array(np.where(row_white == 255))
                data_num_white.append(len(pixels[0]))
            min_pixels = min(data_num_white)
            min_position_value = data_num_white.index(min_pixels)
        
            final_position_value = min_position_value+up_roi    
            row_roi = gray_mask_b[final_position_value, :]
            white_pixels = np.array(np.where(row_roi == 255))
            
            first_white_pixel = int(white_pixels[:,0])
            last_white_pixel = int(white_pixels[:,-1])

            cv2.circle(Image_og, (a_Max[0], final_position_value), 1, (0,192,255), 20)
            cv2.line(Image_og, (a_Max[0], final_position_value), (first_white_pixel, final_position_value), (0, 192, 255), thickness=5)

            cv2.line(Image_og, (first_white_pixel, final_position_value), (last_white_pixel, final_position_value), (0, 0, 192), thickness=5)
            cv2.circle(Image_og,  (first_white_pixel, final_position_value), 1, (0,0,192), 20)
            cv2.circle(Image_og,  (last_white_pixel, final_position_value), 1, (0,0,192), 20)
            
            if IMAGES_DIRECTION == "left_side" :
                Image_og = cv2.flip(Image_og, 1)
                originalImage = cv2.flip(originalImage, 1)
                x,y,w,h = w-230, final_position_value-60, 230, 60
                alpha = 0.3
                cv2.rectangle(Image_og, (x, y), (x + w, y + h), (0,192,255), -1)    
                cv2.putText(Image_og,'Found point', (x + int(w/10),y + int(h/1.5))  ,
                              cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  

            else :  
                x,y,w,h = 0, final_position_value-60, 230, 60
                alpha = 0.3
                cv2.rectangle(Image_og, (x, y), (x + w, y + h), (0,192,255), -1)    
                cv2.putText(Image_og,'Found point', (x + int(w/10),y + int(h/1.5))  ,
                              cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  

            calibration_1_pixel_per_1_cm = 0.03577464788732394
            BF2_distance_pixel = np.sum(row_roi== 255)
            BF2_distance_cm = int((BF2_distance_pixel * calibration_1_pixel_per_1_cm)*10)

            FINAL_GRADE = "A+"+":<20" if BF2_distance_cm <= 20 \
                                    else "A"+":21-24" if BF2_distance_cm <= 24 \
                                    else "B"+":25-28" if BF2_distance_cm <= 28 \
                                    else "C"+":>29" if BF2_distance_cm >= 29 \
                                    else "N/A"
            
            FINAL_GRADE_IMAGES = "A+" if BF2_distance_cm <= 20 \
                                    else "A" if BF2_distance_cm <= 24 \
                                    else "B" if BF2_distance_cm <= 28 \
                                    else "C" if BF2_distance_cm >= 29 \
                                    else "N/A"

        elif  sum(list_condition_upper) == 0 :
            BF2_distance_pixel, BF2_distance_cm = [0] * 2   
            FINAL_GRADE = "N/A"
            FINAL_GRADE_IMAGES = "N/A"
            Image_og = og_images

    except Exception as e:
        BF2_distance_pixel, BF2_distance_cm = [0] * 2
        FINAL_GRADE = "N/A"
        FINAL_GRADE_IMAGES = "N/A"
        Image_og = og_images

    #original_images = originalImage
    return (Image_og ,BF2_distance_pixel,BF2_distance_cm, FINAL_GRADE, FINAL_GRADE_IMAGES)


def analyze_lsq_B_3(Image_og, originalImage, mask_image_B_1, mask_image_B_3, move_x_loin):  

    og_images = Image_og
    maskImage_1 = mask_image_B_1.astype(np.uint8)
    maskImage_3 = mask_image_B_3.astype(np.uint8)

    try :
        gray_og = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
        high, width = gray_og.shape
        image_resize_maskImage_B1 = cv2.resize(maskImage_1, (width, high))
        #image_resize_maskImage_B3 = cv2.resize(maskImage_2, (width, high))
        image_resize_maskImage_B3 = maskImage_3

        first_position_B1 = move_x_loin

        # Select the largest group of section B1 
        coordinate_position_B1, contours_B1 = select_objects(image_resize_maskImage_B1)
        max_hight_index_B1 = max(enumerate(coordinate_position_B1), key=lambda x: x[1][-1])[0]
        position_value_B1 = coordinate_position_B1[max_hight_index_B1]
        mask_image_create_B1 = np.zeros_like(originalImage)
        cv2.drawContours(mask_image_create_B1, contours_B1, max_hight_index_B1, (255,255,255), -1)

        left_img = 0   
        right_img = width
        x1_B1 = position_value_B1[0]
        x2_B1 = x1_B1+position_value_B1[2]

        difference_left = abs(x1_B1 - left_img)
        difference_right = abs(right_img - x2_B1) 

        if difference_left > difference_right :
            IMAGES_DIRECTION = "right_side"
        else :         
            IMAGES_DIRECTION = "left_side"

        # Select the largest group of section B3
        coordinate_position_B3, contour_B3 = select_objects(image_resize_maskImage_B3)
        max_hight_index_B3 = max(enumerate(coordinate_position_B3), key=lambda x: x[1][-1])[0]
        position_value_B3 = coordinate_position_B3[max_hight_index_B3]
        #mask_image_create_B3 = np.zeros_like(originalImage)
        #cv2.drawContours(mask_image_create_B3, contour_B3, max_hight_index_B3, (255,255,255), -1)

        y1_B3 = position_value_B3[1]
        y2_B3 = y1_B3+position_value_B3[3]
        final_position_value = min(y1_B3 ,y2_B3)
        diff_move_og_x_loin  = abs(int(final_position_value - first_position_B1 ))

        gray_originalImage_B1 = cv2.cvtColor(mask_image_create_B1,cv2.COLOR_BGR2GRAY)
        thresh, blackAndWhiteImage_B1 = cv2.threshold(gray_originalImage_B1, 100, 255, cv2.THRESH_BINARY)

        list_data = blackAndWhiteImage_B1[diff_move_og_x_loin,:]

        count_255 = np.count_nonzero(list_data == 255)
        first_white_pixel = np.where(list_data == 255)[0][0] ##### X first
        last_white_pixel = np.where(list_data == 255)[0][-1] ##### X last

        start_point = (first_white_pixel, diff_move_og_x_loin)
        end_point = (last_white_pixel, diff_move_og_x_loin)
        color = (0, 0, 192)

        if IMAGES_DIRECTION == "left_side" :

            cv2.circle(Image_og, (width, diff_move_og_x_loin), 1, (0,192,255), 20)
            cv2.line(Image_og, (width, diff_move_og_x_loin), (first_white_pixel, diff_move_og_x_loin), (0, 192, 255), thickness=5)

            cv2.line(Image_og, (first_white_pixel, diff_move_og_x_loin), (last_white_pixel, diff_move_og_x_loin), (0, 0, 192), thickness=5)
            cv2.circle(Image_og,  (first_white_pixel, diff_move_og_x_loin), 1, (0,0,192), 20)
            cv2.circle(Image_og,  (last_white_pixel, diff_move_og_x_loin), 1, (0,0,192), 20)

            x,y,w,h = width-230, diff_move_og_x_loin-60, 230, 60
            alpha = 0.3
            cv2.rectangle(Image_og, (x, y), (x + w, y + h), (0,192,255), -1)    
            cv2.putText(Image_og,'Found point', (x + int(w/10),y + int(h/1.5))  ,
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  
        else :  

            cv2.circle(Image_og, (0, diff_move_og_x_loin), 1, (0,192,255), 20)
            cv2.line(Image_og, (0, diff_move_og_x_loin), (first_white_pixel, diff_move_og_x_loin), (0, 192, 255), thickness=5)

            cv2.line(Image_og, (first_white_pixel, diff_move_og_x_loin), (last_white_pixel, diff_move_og_x_loin), (0, 0, 192), thickness=5)
            cv2.circle(Image_og,  (first_white_pixel, diff_move_og_x_loin), 1, (0,0,192), 20)
            cv2.circle(Image_og,  (last_white_pixel, diff_move_og_x_loin), 1, (0,0,192), 20)

            x,y,w,h = 0, diff_move_og_x_loin-60, 230, 60
            alpha = 0.3
            cv2.rectangle(Image_og, (x, y), (x + w, y + h), (0,192,255), -1)    
            cv2.putText(Image_og,'Found point', (x + int(w/10),y + int(h/1.5))  ,
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  

        calibration_1_pixel_per_1_cm = 0.03577464788732394
        BF2_distance_pixel = count_255
        #BF2_distance_cm = int((BF2_distance_pixel * calibration_1_pixel_per_1_cm)*10)  
        BF2_distance_cm = count_255  

        FINAL_GRADE = "A+"+":<20" if BF2_distance_cm <= 20 \
                                else "A"+":21-24" if BF2_distance_cm <= 24 \
                                else "B"+":25-28" if BF2_distance_cm <= 28 \
                                else "C"+":>29" if BF2_distance_cm >= 29 \
                                else "N/A"

        FINAL_GRADE_IMAGES = "A+" if BF2_distance_cm <= 20 \
                            else "A" if BF2_distance_cm <= 24 \
                            else "B" if BF2_distance_cm <= 28 \
                            else "C" if BF2_distance_cm >= 29 \
                            else "N/A"
        
    except Exception as e:
            BF2_distance_pixel, BF2_distance_cm = [0] * 2
            FINAL_GRADE = "N/A"
            FINAL_GRADE_IMAGES = "N/A"
            Image_og = og_images

    return (Image_og ,BF2_distance_pixel,BF2_distance_cm, FINAL_GRADE, FINAL_GRADE_IMAGES)

def analyze_lsq_B_4( Image_og, originalImage, mask_image_B_1, mask_image_B_3, move_x_loin, mask_image_c2):  

    originalImage = cv2.resize(originalImage, (originalImage.shape[1]*2, originalImage.shape[0]*2))

    og_images = Image_og
    og_images = cv2.resize(og_images, (og_images.shape[1]*2, og_images.shape[0]*2))

    maskImage_1 = mask_image_B_1.astype(np.uint8)
    maskImage_3 = mask_image_B_3.astype(np.uint8)
    maskImage_c2 = mask_image_c2.astype(np.uint8)

    maskImage_1 = cv2.resize(maskImage_1, (maskImage_1.shape[1]*2, maskImage_1.shape[0]*2))
    maskImage_3 = cv2.resize(maskImage_3, (maskImage_3.shape[1]*2, maskImage_3.shape[0]*2))
    maskImage_c2 = cv2.resize(maskImage_c2, (maskImage_c2.shape[1]*2, maskImage_c2.shape[0]*2))

    try :

        gray_og = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
        high, width = gray_og.shape

        image_resize_maskImage_B1 = cv2.resize(maskImage_1, (width, high))
        image_resize_maskImage_B3 = maskImage_3

        first_position_B1 = move_x_loin*2

        # Select the largest group of section B1 
        coordinate_position_B1, contours_B1 = select_objects(image_resize_maskImage_B1)
        max_hight_index_B1 = max(enumerate(coordinate_position_B1), key=lambda x: x[1][-1])[0]
        position_value_B1 = coordinate_position_B1[max_hight_index_B1]

        left_img = 0   
        right_img = width
        x1_B1 = position_value_B1[0]
        x2_B1 = x1_B1+position_value_B1[2]

        difference_left = abs(x1_B1 - left_img)
        difference_right = abs(right_img - x2_B1) 

        if difference_left > difference_right :
            IMAGES_DIRECTION = "right_side"
        else :         
            IMAGES_DIRECTION = "left_side"

        # Select the largest group of section B3
        coordinate_position_B3, contour_B3 = select_objects(image_resize_maskImage_B3)
        max_hight_index_B3 = max(enumerate(coordinate_position_B3), key=lambda x: x[1][-1])[0]
        position_value_B3 = coordinate_position_B3[max_hight_index_B3]
        mask_image_create_B3 = np.zeros_like(originalImage)
        cv2.drawContours(mask_image_create_B3, contour_B3, max_hight_index_B3, (255,255,255), -1)


        y1_B3 = position_value_B3[1]
        y2_B3 = y1_B3+position_value_B3[3]
        final_position_value = min(y1_B3 ,y2_B3)
        diff_move_og_x_loin = int(width/2)
        diff_move_og_y_loin  = abs(int(final_position_value - first_position_B1 ))
        positionx_see = diff_move_og_x_loin
        positiony_see = diff_move_og_y_loin

        ROI_line1 = abs(positiony_see - 50)
        ROI_line2 = abs(positiony_see + 50)

        if IMAGES_DIRECTION == "right_side" :

            og_images = cv2.flip(og_images, 1)
            originalImage = cv2.flip(originalImage, 1)
            maskImage_1 = cv2.flip(maskImage_1, 1)  
            maskImage_3 = cv2.flip(maskImage_3, 1)  
            maskImage_c2 = cv2.flip(maskImage_c2, 1)  

        elif IMAGES_DIRECTION == "left_side" :
            
            og_images = og_images
            originalImage = originalImage
            maskImage_1 = maskImage_1
            maskImage_3 = maskImage_3
            maskImage_c2 = maskImage_c2

        center_point = (high/2,width/2)

        list_num = []
        list_posi = []

        for roi in range(ROI_line1, ROI_line2):
            positiony_see = roi

            position_1 = abs(positiony_see - 20)
            position_2 = abs(positiony_see + 20)

            ref_see = maskImage_c2[positiony_see,:]
            ref_a = maskImage_c2[position_1,:]
            ref_b = maskImage_c2[position_2,:]

            first_255_see = np.where(ref_see == 255)[0][0]
            first_255_a = np.where(ref_a == 255)[0][0]
            first_255_b = np.where(ref_b >= 250)[0][0]

            x0, y0 = positionx_see,positiony_see
            x1, y1 = first_255_a,position_1
            x2, y2 = first_255_b,position_2
            x3, y3 = first_255_see, positiony_see

            AB = (x1 - x0, y1 - y0)
            BC = (x2 - x1, y2 - y1)
            dot_product = AB[0] * BC[0] + AB[1] * BC[1]
            Length_AB = sqrt((x1 - x0)**2 + (y1 - y0)**2)
            Length_BC = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle_radians = acos(dot_product / (Length_AB * Length_BC))
            angle_degrees = abs( 180 - degrees(angle_radians) )
            
            M = cv2.getRotationMatrix2D(center_point,-(90-angle_degrees),1)

            position1 = rotate_point_wrt_center((x1, y1),(90-angle_degrees),center_point)
            position2 = rotate_point_wrt_center((x2, y2),(90-angle_degrees),center_point)
            position3 = rotate_point_wrt_center((x3, y3),(90-angle_degrees),center_point)
            x_prime1, y_prime1 = position1[0], position1[1] 
            x_prime2, y_prime2 = position2[0], position2[1] 
            x_prime3, y_prime3 = position3[0], position3[1] 
                
            rotated_mask_3 = cv2.warpAffine(maskImage_1,M,(width,high)) 

            ref_see_Rota = rotated_mask_3[y_prime3,:]
            last_position = np.where(ref_see_Rota == 255)[0][-1]

            count_num = np.where(ref_see_Rota > 100)
            num_positions = len(count_num[0])

            last_255_see = last_position
            xfinal, yfinal = last_255_see, y_prime3

            posi1 = rotate_point_wrt_center((x_prime1, y_prime1),-(90-angle_degrees),center_point)
            posi2 = rotate_point_wrt_center((x_prime2, y_prime2),-(90-angle_degrees),center_point)
            posi3 = rotate_point_wrt_center((x_prime3, y_prime3),-(90-angle_degrees),center_point)   
            posifinal = rotate_point_wrt_center((last_255_see, y_prime3),-(90-angle_degrees),center_point)   

            x_old1, y_old1 = posi1[0], posi1[1] 
            x_old2, y_old2 = posi2[0], posi2[1] 
            x_old3, y_old3 = posi3[0], posi3[1] 
            xnewFinal, ynewFinal = posifinal[0], posifinal[1] 

            list_num.append(num_positions)
            list_posi.append([x_old3, y_old3, xnewFinal, ynewFinal])

        min_list_num = min(list_num)
        min_indices = [index for index, value in enumerate(list_num) if value == min_list_num]
        middle_index = len(min_indices) // 2
        middle_value = min_indices[middle_index]
        position_final =  list_posi[middle_value]
                    
        x_old3, y_old3, xnewFinal, ynewFinal = position_final

        cv2.line(og_images, (xnewFinal, ynewFinal), (width, ynewFinal), (0, 192, 255), thickness=5)
        cv2.line(og_images,(x_old3, y_old3),(xnewFinal, ynewFinal),(0,0,192), 5)
        cv2.circle(og_images,(x_old3, y_old3), 1, (0,0,192), 20)
        cv2.circle(og_images,(xnewFinal, ynewFinal), 1, (0,0,192), 20)

        if IMAGES_DIRECTION == "left_side" :

            x,y,w,h = width-230, y_old3-60, 230, 60
            alpha = 0.3
            cv2.rectangle(og_images, (x, y), (x + w, y + h), (0,192,255), -1)    
            cv2.putText(og_images,'Found point', (x + int(w/10),y + int(h/1.5))  ,
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  
        else :  

            og_images = cv2.flip(og_images, 1)

            x,y,w,h = 0, y_old3-60, 230, 60
            alpha = 0.3
            cv2.rectangle(og_images, (x, y), (x + w, y + h), (0,192,255), -1)    
            cv2.putText(og_images,'Found point', (x + int(w/10),y + int(h/1.5))  ,
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  

        og_images = cv2.resize(og_images, (int(og_images.shape[1]/2), int(og_images.shape[0]/2)))
        Image_og = og_images

        count_255 = min_list_num

        calibration_1_pixel_per_1_cm = 0.03577464788732394
        BF2_distance_pixel = count_255
        #BF2_distance_cm = int((BF2_distance_pixel * calibration_1_pixel_per_1_cm)*10)  
        BF2_distance_cm = count_255  

        FINAL_GRADE = "A+"+":<20" if BF2_distance_cm <= 20 \
                                else "A"+":21-24" if BF2_distance_cm <= 24 \
                                else "B"+":25-28" if BF2_distance_cm <= 28 \
                                else "C"+":>29" if BF2_distance_cm >= 29 \
                                else "N/A"

        FINAL_GRADE_IMAGES = "A+" if BF2_distance_cm <= 20 \
                            else "A" if BF2_distance_cm <= 24 \
                            else "B" if BF2_distance_cm <= 28 \
                            else "C" if BF2_distance_cm >= 29 \
                            else "N/A"

        calibration_1_pixel_per_1_cm = 0.03577464788732394
        BF2_distance_pixel = count_255
        #BF2_distance_cm = int((BF2_distance_pixel * calibration_1_pixel_per_1_cm)*10)  
        BF2_distance_cm = count_255  

        FINAL_GRADE = "A+"+":<20" if BF2_distance_cm <= 20 \
                                else "A"+":21-24" if BF2_distance_cm <= 24 \
                                else "B"+":25-28" if BF2_distance_cm <= 28 \
                                else "C"+":>29" if BF2_distance_cm >= 29 \
                                else "N/A"

        FINAL_GRADE_IMAGES = "A+" if BF2_distance_cm <= 20 \
                            else "A" if BF2_distance_cm <= 24 \
                            else "B" if BF2_distance_cm <= 28 \
                            else "C" if BF2_distance_cm >= 29 \
                            else "N/A"

    except Exception as e:

        if IMAGES_DIRECTION == "right_side" :
            og_images = cv2.flip(og_images, 1)

        elif IMAGES_DIRECTION == "left_side" :            
            og_images = og_images

        BF2_distance_pixel, BF2_distance_cm = [0] * 2
        FINAL_GRADE = "N/A"
        FINAL_GRADE_IMAGES = "N/A"

        og_images = cv2.resize(og_images, (int(og_images.shape[1]/2), int(og_images.shape[0]/2)))
        Image_og = og_images


    return (Image_og ,BF2_distance_pixel,BF2_distance_cm, FINAL_GRADE, FINAL_GRADE_IMAGES)




