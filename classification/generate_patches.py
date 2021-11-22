import openslide
import xmltodict
import numpy as np
from PIL import Image
import os
import cv2
import json
import sys
import matplotlib.pyplot as plt


def read_mask(simg, xml_file, output_dir):
    # read region
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    layers = doc['Annotations']['Annotation']

    start_x, start_y = get_nonblack_starting_point(simg)
    patches = []

    try:
        multi_contours = layers['Regions']

        if len(multi_contours) < 2:
            notFound = multi_contours[0]

        else:
            contours = multi_contours['Region']

            try:
                contours['Vertices']
                img, cimg, mask, bbox = get_contour(simg, contours, start_x, start_y)
                patch_fname = save_patch(output_dir, xml_file, img, bbox, idx=0)
                patches.append(patch_fname)
                print('case 1')

            except:
                for j in range(len(contours)):
                    contour = contours[j]
                    img, cimg, mask, bbox = get_contour(simg, contour, start_x, start_y)
                    patch_fname = save_patch(output_dir, xml_file, img, bbox, idx=j)
                    patches.append(patch_fname)

    except:
        for i in range(len(layers)):
            contours = layers[i]['Regions']

            if len(contours) < 2:
                notFound = layers[0]
                print('case 2')

            else:
                contours = contours['Region']

                try:
                    contours['Vertices']
                    img, cimg, mask, bbox = get_contour(simg, contours, start_x, start_y)
                    patch_fname = save_patch(output_dir, xml_file, img, bbox, idx=0)
                    patches.append(patch_fname)
                    print('case 3.1')

                except:
                    for j in range(len(contours)):
                        contour = contours[j]
                        img, cimg, mask, bbox = get_contour(simg, contour, start_x, start_y)
                        patch_fname = save_patch(output_dir, xml_file, img, bbox, idx=j)
                        patches.append(patch_fname)
                        print('case 3.2')

    return patches


def save_patch(output_dir, xml_file, img, bbox, idx=0):
    img_all_out = Image.fromarray(img)
    img_all_out = img_all_out.resize((256, 256))
    img_all_out_file = os.path.join(output_dir, '%s-x-ROI_%d-x-%d-x-%d-x-%d-x-%d.png' %
                                    (os.path.basename(xml_file).strip('.xml'), idx, bbox[0], bbox[1], bbox[2], bbox[3]))
    img_all_out.save(img_all_out_file)
    print(idx, img_all_out_file)

    return img_all_out_file


def get_none_zero(black_arr):
    nonzeros = black_arr.nonzero()
    starting_y = nonzeros[0].min()
    ending_y = nonzeros[0].max()
    starting_x = nonzeros[1].min()
    ending_x = nonzeros[1].max()

    return starting_x, starting_y, ending_x, ending_y


def scan_nonblack(simg, px_start, py_start, px_end, py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_start+offset_x, py_start), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start+offset_x, py_start), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_start, py_start+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start, py_start+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_start+offset_x-1
    y = py_start+offset_y-1
    return x, y


def get_nonblack_starting_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    # staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    # ending point
    px3 = (ending_x + 1) * multiples
    py3 = (ending_y + 1) * multiples

    xx, yy = scan_nonblack(simg, px2, py2, px3, py3)

    return xx, yy


def get_contour(simg, contour, start_x, start_y):
    try:
        max_height = int(simg.properties['openslide.bounds-width'])
        max_widths = int(simg.properties['openslide.bounds-height'])
    except:
        max_height = int(simg.properties['aperio.OriginalWidth'])
        max_widths = int(simg.properties['aperio.OriginalHeight'])
    vertices = contour['Vertices']['Vertex']
    x_min = max_height
    x_max = 0
    y_min = max_widths
    y_max = 0


    # xmin = int(round(min(float(vertices[0]['@Y']),float(vertices[1]['@Y']))))
    # xmax = int(round(max(float(vertices[0]['@Y']),float(vertices[1]['@Y']))))
    # ymin = int(round(min(float(vertices[0]['@X']),float(vertices[1]['@X']))))
    # ymax = int(round(max(float(vertices[0]['@X']),float(vertices[1]['@X']))))
    #
    # xx_min = xmin - 50
    # xx_max = xmax + 50
    # yy_min = ymin - 50
    # yy_max = ymax + 50
    #
    # heights = xx_max - xx_min
    # widths = yy_max - yy_min
    #
    # xs = xx_min
    # yss = yy_max

    for vi in range(len(vertices)):
        xraw = float(vertices[vi]['@X'])
        if 'leica.device-model' in simg.properties:
            yraw = float(vertices[vi]['@Y'])
        elif 'aperio.Filename' in simg.properties:
            yraw = max_widths-float(vertices[vi]['@Y'])
        if xraw < x_min:
            x_min = xraw
        if xraw > x_max:
            x_max = xraw
        if yraw < y_min:
            y_min = yraw
        if yraw > y_max:
            y_max = yraw

        x_min = int(round(x_min))
        x_max = int(round(x_max))
        y_min = int(round(y_min))
        y_max = int(round(y_max))

        # add cropping
        xx_min = max(x_min-50, 0)
        xx_max = min(x_max+50, max_height)
        yy_min = max(y_min-50, 0)
        yy_max = min(y_max+50, max_widths)

        heights = xx_max-xx_min
        widths = yy_max-yy_min

        xs = xx_min
        yss = yy_max


    cnt = np.zeros((len(vertices),1,2))
    for vi in range(len(vertices)):
        xx = float(vertices[vi]['@X'])-xs
        yy = yss - float(vertices[vi]['@Y'])
        cnt[vi,0,0] = int(xx)
        cnt[vi,0,1] = int(yy)

    read_x0 = start_x+xs
    read_y0 = max_widths-yss+start_y
    read_height = heights
    read_widths = widths
    bbox = (read_x0, read_y0, read_height, read_widths)

    img = simg.read_region((read_x0, read_y0), 0, (read_height,read_widths))
    img = np.array(img.convert('RGB'))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cimg = img.copy()
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((len(vertices),1,2))
    for vi in range(len(vertices)):
        xx = float(vertices[vi]['@X'])-xs
        yy = yss - float(vertices[vi]['@Y'])
        cnt[vi,0,0] = int(xx)
        cnt[vi,0,1] = int(yy)

    cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 3)

    # draw mask
    mask = np.zeros(cimg.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt.astype(int)], -1, (255, 255, 255), -1)

    return img, cimg, mask, bbox


if __name__ == "__main__":
    scn_file = sys.argv[1]
    case_dir = sys.argv[2]

    case = os.path.basename(case_dir)
    det_xml = os.path.join(case_dir,  f'{case}.xml')

    try:
        simg = openslide.open_slide(scn_file)

    except:
        print('%s', case)
        exit()

    output_dir = os.path.join(case_dir, 'patch')
    os.makedirs(output_dir, exist_ok=True)
    patch_files = read_mask(simg, det_xml, output_dir)

    patch_files_lst = [{'image_dir': os.path.abspath(img), 'target': -1} for img in patch_files]

    with open(f'{case_dir}/patch.json', 'w') as f:
        json.dump(patch_files_lst, f)