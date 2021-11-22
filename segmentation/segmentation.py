import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import xlsxwriter
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from plantcv import plantcv as pcv
from utils.testset import BasicDataset
from torch.utils.data import DataLoader
import xmltodict
import json
import cv2


def run_one_wsi(rootdir, wsi, model):
    print('Start segmentation...')
    wsi_name = os.path.splitext(os.path.basename(wsi))[0]
    wsi_path = os.path.join(rootdir,wsi_name)
    patch_class = ['normal','obsolescent','solidified','disappearing']
    # patch_class = ['test']
    xml_file = os.path.join(rootdir,wsi_name,wsi_name+'.xml')
    pred_file = os.path.join(rootdir,wsi_name,'patch_pred_5c.json')
    simg = openslide.open_slide(wsi)


    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())

    with open(pred_file) as f:
        result = json.load(f)

    # non_glo_index = next((index for (index, d) in enumerate(result) if d["pred"] == 4), None)
    non_glo_index = [i for i,_ in enumerate(result) if _['pred'] == 4]

    bbox = doc['Annotations']['Annotation']['Regions']['Region']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101'
                         , pretrained=False, num_classes=2)

    net.load_state_dict(torch.load(model, map_location=device))
    net.cuda()
    net.eval()

    xlist = []
    ylist = []

    for index in range(len(patch_class)):

        img_path = os.path.join(wsi_path,'patch',patch_class[index])

        datatest = BasicDataset(img_path, 1, 'RGB', 'test')
        if len(datatest) == 0 :
            break

        test_loader = DataLoader(datatest, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                                  drop_last=False)

        out_dir = os.path.join(rootdir ,wsi_name, 'seg_output' , patch_class[index])
        mask_check_dir  = os.path.join(rootdir ,wsi_name, 'mask_check' , patch_class[index])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        tbar = tqdm(test_loader, desc='\r')

        if not os.path.exists(mask_check_dir):
            os.makedirs(mask_check_dir)

        for idx, batch in enumerate(tbar):
            tbar.set_description(patch_class[index])

            imgs, filename = batch['image'], batch['filename']
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_pred = np.zeros(1)
            with torch.no_grad():
                mask_pred = net(imgs)

            mask_pred = mask_pred['out']
            pred = mask_pred.max(dim=1)[1]
            pred = (pred).float()

            output_seg = mask_pred.max(dim=1)[1].unsqueeze(1)
            output_seg = output_seg.data.cpu().numpy()


            for i in range(output_seg.shape[0]):
                output_img = output_seg[i, 0, :, :] * 255

                # width, height = output_img.shape # Get dimensions
                width = 384
                height = 384
                new_width = 256
                new_height = 256
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2
                id = int(filename[i].split('-x-')[1].split('_')[1])


                filepath = os.path.join(out_dir, filename[i] + ".png")
                mask = Image.fromarray(output_img.astype(np.uint8)).resize((width,height)).crop((left, top, right, bottom))
                new_mask_array = np.array(mask)
                ret, thresh = cv2.threshold(new_mask_array, 50, 255, cv2.THRESH_BINARY)
                _, contours ,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = find_largest_contour(contours)

                img = imgs[i].cpu().detach().numpy()
                img = np.transpose(img , (1, 2, 0))
                # img = img.swapaxes(0, 1)
                # img = img.swapaxes(1, 2)
                img = Image.fromarray((img * 255).astype(np.uint8)).resize((width,height)).crop((left, top, right, bottom))
                img = np.asarray(img)
                mask_deb = cv2.drawContours(img, contours, 0, (0, 255, 0), 2)
                M = cv2.moments(contours[0])
                center_x, center_y = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])
                xlist.append(center_x)
                ylist.append(center_y)
                cv2.circle(mask_deb, (center_x, center_y), 5, (0, 255, 0), -1)
                mask = pcv.roi.roi2mask(img, contours)


                cv2.imwrite(os.path.join(mask_check_dir, filename[i] + ".png"),cv2.cvtColor(mask_deb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(filepath,mask)
                ratio = sum(sum(new_mask_array/255))/(256**2)

                   # Image.fromarray(output_img.astype(np.uint8)).resize(( 256,256)).save(filepath)
                id = int(filename[i].split('-x-')[1].split('_')[1])
                name = result[id]['image_dir']
                x1 = float(bbox[id]['Vertices']['Vertex'][0]['@X'])
                x2 = float(bbox[id]['Vertices']['Vertex'][1]['@X'])
                y1 = float(bbox[id]['Vertices']['Vertex'][0]['@Y'])
                y2 = float(bbox[id]['Vertices']['Vertex'][1]['@Y'])
                start_x = min(x1,x2)
                start_y = min(y1,y2)
                length = int(abs(x2-x1)-100)
                num_of_pix = length**2    #add 50 to each edge when cutting patches
                glo_pix = int(ratio*num_of_pix)
                score = bbox[id]['@Text']
                result[id]['class'] = patch_class[result[id]['pred']]
                result[id]['nun_of_pix'] = glo_pix
                result[id]['area'] = glo_pix * float(simg.properties['openslide.mpp-x']) * float(simg.properties['openslide.mpp-y'])
                result[id]['probabilty'] = score
                result[id]['center_x'] = start_x + length/2
                result[id]['center_y'] = start_y + length/2

    result = popbyindex(result,non_glo_index)

    output_f = os.path.join(rootdir,wsi_name,'patch_area_prob.json')
    with open(output_f, 'w') as f:
        json.dump(result, f)

    for i in range(len(result)):
        result[i]['image_dir'] = os.path.basename(result[i]['image_dir'])

    nor_list = [result[index] for index in [i for i,_ in enumerate(result) if _['pred'] == 0]]
    obs_list = [result[index] for index in [i for i,_ in enumerate(result) if _['pred'] == 1]]
    sol_list = [result[index] for index in [i for i,_ in enumerate(result) if _['pred'] == 2]]
    dis_list = [result[index] for index in [i for i,_ in enumerate(result) if _['pred'] == 3]]

    with pd.ExcelWriter(os.path.join(rootdir,wsi_name,'result.xlsx')) as writer:
        df1 = pd.DataFrame.from_dict(nor_list)
        df2 = pd.DataFrame.from_dict(obs_list)
        df3 = pd.DataFrame.from_dict(sol_list)
        df4 = pd.DataFrame.from_dict(dis_list)
        df1.to_excel(writer, sheet_name='normal')
        df2.to_excel(writer, sheet_name='obsolescent')
        df3.to_excel(writer, sheet_name='solidified')
        df4.to_excel(writer, sheet_name='disappearing')










def find_largest_contour(contourlist):
    current = 0
    index = 0
    for i in range(len(contourlist)):
        if contourlist[i].shape[0] >current:
            current =  contourlist[i].shape[0]
            index = i

    new_contour = [contourlist[index]]

    return new_contour

def popbyindex(originallist,indexlist):
    for index in sorted(indexlist, reverse = True):
        del originallist[index]

    return originallist




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str)
    parser.add_argument('--wsi', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()


    run_one_wsi(args.rootdir, args.wsi, args.model)


