import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

root = 'imagenet/train/'
bbox_root = 'imagenet/bbox_select_txt/'
save_root = 'ImageNet291/'
mapping_path = 'ImageNet_To_ImageNet291.txt'


if __name__ == "__main__":
    
    f = open(mapping_path, 'r')
    mappings = f.readlines()
    f.close()
    progress = 0
    for line in tqdm(mappings):
        if '.jpg' not in line:
            progress += 1
            print('%03d/291'%(progress))
            foldername = line[:-1]
            with open(os.path.join(bbox_root, f'{foldername}.bbox.txt'), 'r') as f:
                bbxlines = f.read().splitlines()
            bboxdict = {}
            for l in bbxlines:
                bboxdict[l.split()[0]] = l.split()[1:]
            trymakedir = True
            continue
        img_name, save_name = line.split()
        if trymakedir:
            os.makedirs(os.path.join(save_root, 'train', save_name.split('/')[1]), exist_ok=True)
            os.makedirs(os.path.join(save_root, 'test', save_name.split('/')[1]), exist_ok=True)
            trymakedir = False

        bbox = bboxdict[img_name]
        x, y, w, h = [round(float(bbox[0])), round(float(bbox[1])), round(float(bbox[2])), round(float(bbox[3]))]

        imgpil = Image.open(os.path.join(root, foldername, img_name)).convert('RGB')
        img = np.array(imgpil)
        H, W, C = img.shape
        halfy = round(y + h / 2)
        halfx = round(x + w / 2)
        R = min(min(max(max((h * w * 4 / 3.) ** 0.5, h), w), H), W)
        Rx = R if R >= w else min(R * 4 / 3, w)
        Ry = R if R >= h else min(R * 4 / 3, h)
        halfRx = round(Rx / 2)
        Rx = round(Rx)
        halfRy = round(Ry / 2)
        Ry = round(Ry)
        offsety = max(0, halfRy - halfy)
        if offsety == 0:
            offsety = min(H - halfy - halfRy, 0)
        offsetx = max(0, halfRx - halfx)
        if offsetx == 0:
            offsetx = min(W - halfx - halfRx, 0)
        x_ = max(0, halfx - halfRx + offsetx)
        y_ = max(0, halfy - halfRy + offsety)
        subimg = img[y_:min(y_ + Ry, H), x_:min(x_ + Rx, W), :]
        try:
            subimgpil = Image.fromarray(subimg)
        except:
            continue
        subimgpil = subimgpil.resize((256, 256), Image.BILINEAR)
        subimgpil.save(os.path.join(save_root, save_name))
