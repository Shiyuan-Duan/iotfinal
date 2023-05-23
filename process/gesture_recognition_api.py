import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from .model import *
from .my_network import *
import cv2

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(123)

def gesture_recognition(img, model_path):
    '''
    img: np.array, model_path: the saved model parameters
    '''

    gesture_list = ['none', 'play', 'stop', 'vol-down', 'vol-up']
    img_size = 96

    img_transforms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ])

    # data preparation
    img = Image.fromarray(img)
    img = img_transforms(img)

    # Initializing a pretrained model, customizing the final classifier layer, and freezing all but the final layer  
    model = resnet18_cbam()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, model.fc.in_features),
                                    nn.Linear(model.fc.in_features, 5))
    trained_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(trained_state_dict)
    # model = model.cuda()
    model.eval()

    # img = img.cuda()
    img = img.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, prediction = torch.max(outputs, 1)
    
    res = gesture_list[prediction]
    print(res)
    return res

## for testing
if __name__ == '__main__':
    img = cv2.imread('./images/val/play/2023-05-22_16-33-10.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)

    res = gesture_recognition(img, 'gesture_model.pt')