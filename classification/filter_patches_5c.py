import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import os
import json
import models.resnetv2 as resnetv2
import sys
from data_loader import DataLoader



def classification(checkpoint, data):


    print("==> loading model")
    num_classes = 5
    model = resnetv2.KNOWN_MODELS['BiT-M-R101x1'](head_size=num_classes, zero_head=True)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print('Load pre-trained model')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("==> start filtering")
    filter_patches(model, data, transform)


def filter_patches(model, input_data, transform):
    dataset = DataLoader(input_data, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    tbar = tqdm(data_loader, desc='\r')


    model.to(device)
    model.eval()

    with torch.no_grad():
        probs = []
        preds = []
        paths = []

        for idx, (inputs, _, image_path) in enumerate(tbar):
            tbar.set_description('lesion characterization')
            inputs = inputs.to(device)

            outputs = model(inputs).to(device)
            prob = F.softmax(outputs, dim=1)

            probs.extend(prob.tolist())
            preds.extend(torch.argmax(prob, dim=1).tolist())
            paths.extend(list(image_path))



        binary_preds = map(lambda x: 0 if x != 4 else 1, preds)

    result = [{'image_dir': img, 'pred': pred} for img, pred in zip(paths, binary_preds)]
    result_5c = [{'image_dir': img, 'pred': pred} for img, pred in zip(paths, preds)]

    # output_f = os.path.join(os.path.dirname(input_data), 'patch_pred.json')
    # with open(output_f, 'w') as f:
    #     json.dump(result, f)
    output_f = os.path.join(os.path.dirname(input_data), 'patch_pred.json')
    with open(output_f, 'w') as f:
        json.dump(result, f)


    output_f = os.path.join(os.path.dirname(input_data), 'patch_pred_5c.json')
    with open(output_f, 'w') as f:
        json.dump(result_5c, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='best model')
    parser.add_argument('--case', required=True,  help='case name')
    parser.add_argument('--output_dir', required=True, help='case name')
    opt = parser.parse_args()
    case = os.path.splitext(os.path.basename(opt.case))[0]
    data = os.path.join(opt.output_dir, case, 'patch.json')
    classification(opt.checkpoint, data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



