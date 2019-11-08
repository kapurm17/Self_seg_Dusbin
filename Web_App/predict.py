from fastai.vision import *
from fastai.metrics import error_rate
import pandas as pd
import numpy as np
from torchvision import transforms, models
import torch
import PIL
import io

def recreate_model(weights_path):
    '''
    '''
    weights = torch.load(weights_path)
    body = create_body(models.resnet34, True, None)
    class_num = 6
    nf = callbacks.hooks.num_features_model(body) * 2
    head = create_head(nf, class_num, None, ps=0.5, bn_final=False)
    model = nn.Sequential(body, head)
    model.eval()
    model.load_state_dict(weights['model'])
    return model

def get_tensor(img_bytes):
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])
    image = PIL.Image.open(io.BytesIO(img_bytes))
    return test_transforms(image).usqueeze_(0)



def predict_img(model, img_path):
    '''
    '''
    data_class = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    img = PIL.Image.open(img_path)
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])
    image_tensor = test_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return data_class[index]

def make_predictions(weights_path, img_path):
    '''
    '''
    model= recreate_model(weights_path)
    return predict_img(model, img_path)



if  __name__=='__main__':
    weights_path = "../weights.pth"
    model=recreate_model(weights_path)
    img_path="../cardboard59.jpg"
    prediction = predict_img(model, img_path)
    print(prediction)
