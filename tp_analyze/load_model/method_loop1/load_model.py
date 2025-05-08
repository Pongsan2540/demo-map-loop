from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import torch

def load_sentence(name_model_trans, device):
    model_sentence = SentenceTransformer(name_model_trans)
    return model_sentence

def load_blip(name_model_vector, device):
    processor = BlipProcessor.from_pretrained(name_model_vector, use_fast=True)
    generation = BlipForConditionalGeneration.from_pretrained(name_model_vector)
    model_blip = (processor, generation)
    return model_blip

def load_yolo(name_model_detect, device):
    model_detect = YOLO('./model/model_yolo/' + name_model_detect) 

    return model_detect

def main_load(list_model):

    name_model_trans, name_model_vector, name_model_detect = list_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sentence = load_sentence(name_model_trans, device)
    model_blip = load_blip(name_model_vector, device)
    model_yolo = load_yolo(name_model_detect, device)

    return (model_sentence, model_blip, model_yolo, device)