import xmltodict
import json
import os
import sys
import shutil
from xml.etree import ElementTree as ET






root_dir = sys.argv[1]
case = sys.argv[2]
xml_file = f'{root_dir}/{case}.xml'
pred_file = f'{root_dir}/patch_pred_5c.json'




with open(xml_file) as fd:
    doc = xmltodict.parse(fd.read())


with open(pred_file) as f:
    pred = json.load(f)



glom_dir = root_dir + '/patch/normal/'
non_glom_dir = root_dir + '/patch/non_glom/'
dis_dir = root_dir + '/patch/disappearing/'
obs_dir = root_dir + '/patch/obsolescent/'
sol_dir = root_dir + '/patch/solidified/'
os.makedirs(glom_dir, exist_ok=True)
os.makedirs(non_glom_dir, exist_ok=True)
os.makedirs(dis_dir, exist_ok=True)
os.makedirs(obs_dir, exist_ok=True)
os.makedirs(sol_dir, exist_ok=True)


# for files in os.listdir(glom_dir):
#     shutil.move(files,root_dir)
# for files in os.listdir(non_glom_dir):
#     shutil.move(files,root_dir)


for i in range(len(pred)):
    if pred[i]['pred'] == 0:
        shutil.move(pred[i]['image_dir'], glom_dir)
        pred[i]['image_dir'] = pred[i]['image_dir'].replace('patch', 'patch/normal')
    elif pred[i]['pred'] == 1:
        shutil.move(pred[i]['image_dir'], obs_dir)
        pred[i]['image_dir'] = pred[i]['image_dir'].replace('patch', 'patch/obsolescnet')
    elif pred[i]['pred'] == 2:
        shutil.move(pred[i]['image_dir'], sol_dir)
        pred[i]['image_dir'] = pred[i]['image_dir'].replace('patch', 'patch/solidified')
    elif pred[i]['pred'] == 3:
        shutil.move(pred[i]['image_dir'], dis_dir)
        pred[i]['image_dir'] = pred[i]['image_dir'].replace('patch', 'patch/disappearing')
    elif pred[i]['pred'] == 4:
        shutil.move(pred[i]['image_dir'], non_glom_dir)
        pred[i]['image_dir'] = pred[i]['image_dir'].replace('patch', 'patch/non_glom')

    output = os.path.join(root_dir, 'patch_pred_5c.json')
    with open(output, 'w') as f:
        json.dump(pred, f)





det_patches = doc['Annotations']['Annotation']['Regions']['Region']

non_glom = [i for i in pred if i['pred'] == 4]
non_glom_idx = [int(os.path.basename(i['image_dir']).split('-x-')[1].split('_')[1]) for i in non_glom]
non_glom_id = [str(i + 1) for i in non_glom_idx]

glom = [i for i in pred if i['pred'] == 0]
glom_idx = [int(os.path.basename(i['image_dir']).split('-x-')[1].split('_')[1]) for i in glom]
glom_id = [str(i + 1) for i in glom_idx]

obs = [i for i in pred if i['pred'] == 1]
obs_idx = [int(os.path.basename(i['image_dir']).split('-x-')[1].split('_')[1]) for i in obs]
obs_id = [str(i + 1) for i in obs_idx]

sol = [i for i in pred if i['pred'] == 2]
sol_idx = [int(os.path.basename(i['image_dir']).split('-x-')[1].split('_')[1]) for i in sol]
sol_id = [str(i + 1) for i in sol_idx]

dis = [i for i in pred if i['pred'] == 3]
dis_idx = [int(os.path.basename(i['image_dir']).split('-x-')[1].split('_')[1]) for i in dis]
dis_id = [str(i + 1) for i in dis_idx]

# print(f'{len(det_patches)} detected glomeruli, {len(det_patches)  -  len(non_glom_id)} filtered glomeruli')

glom = []
for idx, patch in enumerate(det_patches):
    if patch['@Id']  in glom_id:
        glom_patch = patch.copy()
        glom_patch['@Id'] = len(glom) + 1
        glom.append(glom_patch)

obs = []
for idx, patch in enumerate(det_patches):
    if patch['@Id'] in obs_id:
        obs_patch = patch.copy()
        obs_patch['@Id'] = len(obs) + 1
        obs.append(obs_patch)
sol = []
for idx, patch in enumerate(det_patches):
    if patch['@Id'] in sol_id:
        sol_patch = patch.copy()
        sol_patch['@Id'] = len(sol) + 1
dis = []
for idx, patch in enumerate(det_patches):
    if patch['@Id'] in dis_id:
        dis_patch = patch.copy()
        dis_patch['@Id'] = len(dis) + 1
        dis.append(dis_patch)

root = ET.parse(xml_file).getroot()
class2 = ET.parse(xml_file).getroot()
class3 = ET.parse(xml_file).getroot()
class4 = ET.parse(xml_file).getroot()

add = class2[0]
root.append(add)
add = class3[0]
root.append(add)
add = class4[0]
root.append(add)

root[0].attrib['Id'],root[0].attrib['Name'],root[0].attrib['LineColor'] = '1','normal','255'
root[1].attrib['Id'],root[1].attrib['Name'],root[1].attrib['LineColor'] = '2','obsolescent','65535'
root[2].attrib['Id'],root[2].attrib['Name'],root[2].attrib['LineColor'] = '3','solidified','16744448'
root[3].attrib['Id'],root[3].attrib['Name'],root[3].attrib['LineColor'] = '4','disappearing','65280'

out = ET.ElementTree(root)
out.write(f'{root_dir}/temp.xml', encoding="utf-8", xml_declaration=True)

with open(f'{root_dir}/temp.xml') as fd:
    data = xmltodict.parse(fd.read())

data['Annotations']['Annotation'][0]['Regions']['Region'] = glom
data['Annotations']['Annotation'][1]['Regions']['Region'] = obs
data['Annotations']['Annotation'][2]['Regions']['Region'] = sol
data['Annotations']['Annotation'][3]['Regions']['Region'] = dis


out = xmltodict.unparse(data, pretty=True)
xml_file = f'{root_dir}/ftd_patch_5c.xml'
with open(xml_file, 'wb') as file:
    file.write(out.encode('utf-8'))

with open(f'{root_dir}/result.txt', 'w') as f:
    content0 = 'normal:' + ' ' + str(len(glom)) + '\r\n'
    content1 = 'obsolescent:' + ' ' + str(len(obs)) + '\r\n'
    content2 = 'solidified:' + ' ' + str(len(sol)) + '\r\n'
    content3 = 'disappearing:' + ' ' + str(len(dis)) + '\r\n'
    content4 = 'non_glom:' + ' ' + str(len(non_glom)) + '\r\n'
    f.write(content0+content1+content2+content3+content4)
