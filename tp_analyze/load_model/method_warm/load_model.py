import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import segmentation_models_pytorch as smp
from torchvision import transforms as T


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

def load_model_sam(model_sam, device):
    model_mapping = {
                    'sam_vit_h_4b8939.pth': 'vit_h',
                    'sam_vit_l_0b3195.pth': 'vit_l',
                    'sam_vit_b_01ec64.pth': 'vit_b'
                    }
    model_type = model_mapping.get(model_sam)
    sam_checkpoint = './model/model_sam/'+model_sam
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor_sam = SamPredictor(sam)
    return predictor_sam

def load_model_unet(model_unet, device):

    path_model_unet_2 = './model/model_unet/'+model_unet
    model_unet_1 = torch.load(path_model_unet_2, map_location=device)
    model_unet_1 = model_unet_1.to(device)

    per_imges = np.zeros((1152, 768, 3), dtype = np.uint8)
    test_predict_image_mask_miou(model_unet_1, per_imges, device)
    return model_unet_1

def main_load(list_model):

    model1, model2, model3 = list_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_unet1 = load_model_unet(model1, device)
    model_unet2 = load_model_unet(model2, device)
    model_sam = load_model_sam(model3, device)

    return (model_unet1, model_unet2, model_sam, device)