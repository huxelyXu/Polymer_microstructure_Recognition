import argparse
import torch
from torchvision import models, transforms, utils
import numpy as np
from model.mynet import mynet_light
from src.guidedBackProp import *
from src.smoothGrad import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def run(image_path, index, cuda):
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (101, 101))
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image).unsqueeze(0)

    model = mynet_light()
    model.load_state_dict(
        torch.load(r'../data/4Vi/visual.ckpt', map_location=torch.device(device))['state_dict'])

    # guidedBackProp
    print("guidedBackProp...")
    guided_bp = GuidedBackProp(model, use_cuda=False)
    guided_cam, _ = guided_bp(image)
    cv2.imwrite("results/g/guidedbackProp_" + image_path.split('/')
    [-1], arrange_img(guided_cam))

    # smoothGrad
    print("smoothGrad...")
    smooth_grad = SmoothGrad(model, use_cuda=False, stdev_spread=0.2, n_samples=20)
    smooth_cam, _ = smooth_grad(image)
    cv2.imwrite("results/s/smoothGrad_" + image_path.split('/')
    [-1], show_as_gray_image(smooth_cam))



if __name__ == '__main__':
    path = r'../data/4Vi/data_all/'
    txtpath = r'../data/4Vi/data_all/train.txt'
    datainfo = open(txtpath, 'r')
    img = []
    for line in datainfo:
        li = line.strip('\n')
        if li == 'train.bat' or li == 'train.txt':
            continue
        img.append(li)
    for filename in img:
        words = filename.split('_')
        leibie = words[0]
        parser = argparse.ArgumentParser(description='guidedBP, smoothGrad')
        parser.add_argument('--image_path', default=path + filename, required=False)
        parser.add_argument('--cuda', action='store_true', required=False)
        parser.add_argument('--index', type=int, default=leibie, required=False)
        args = parser.parse_args()
        run(args.image_path, args.index, args.cuda)


