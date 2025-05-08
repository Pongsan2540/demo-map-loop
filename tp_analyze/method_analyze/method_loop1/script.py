################################################################################
# script python of API warmApi version date 20240108
#               - add analyze_lsq_B_3
################################################################################

from minio import Minio
from minio.error import S3Error
from io import BytesIO
import io
from datetime import datetime
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def minio_upload(image, ip_minIO, name_key, password_key, bucket_name, image_name):

    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = io.BytesIO(buffer)

    minio_client = Minio(
                            ip_minIO,
                            access_key=name_key,
                            secret_key=password_key,
                            secure=False  # เปลี่ยนเป็น True หากใช้ HTTPS
                        )

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    minio_client.put_object(
                                bucket_name,
                                image_name,
                                image_bytes,
                                length=len(buffer),
                                content_type="image/jpeg"
                            )

    link_image = "http://"+ip_minIO+"/"+bucket_name+"/"+image_name
    return link_image

def extract_object_color(image, model, bbox):

    processor, model_Blip = model
    
    x1, y1, x2, y2 = map(int, bbox) 
    roi = image[y1:y2, x1:x2] 

    #roi_base64 = base64.b64encode(cv2.imencode('.jpeg', roi)[1]).decode()
    #image_data_base64 = base64.b64encode(cv2.imencode('.jpeg', roi,[cv2.IMWRITE_JPEG_QUALITY, 50])[1]).decode()  
    #roi_base64 = (f"data:image/jpeg;base64,{image_data_base64}")

    inputs = processor(images=roi, return_tensors="pt")
    out = model_Blip.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption