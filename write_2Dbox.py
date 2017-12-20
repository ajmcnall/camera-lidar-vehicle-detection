#! /usr/bin/python3
import os
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import PIL
import numpy as np
import argparse

def rot(n, theta):
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def get_bbox(p0, p1):

    v = np.array([[p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                  [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                  [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
                  [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)

    return v, e


parser = argparse.ArgumentParser(description="Rewrite image and boxes for training")
parser.add_argument('--data_path', dest='datapath',
                        help='data path of images and 3D boxes',
                        default='rob599_dataset_deploy/trainval', type=str)
parser.add_argument('--save_path', dest='savepath',
                        help='save images and corresponding 2D boxes',
                        default='./car_images/', type=str)

args = parser.parse_args() 
files = glob(os.path.join(args.datapath, '*/*_image.jpg'))
savepath = args.savepath

# files = glob('data/rob599_dataset_deploy/trainval/*/*_image.jpg')
# savepath = './data/car_images/'
if not os.path.exists(savepath):
        os.makedirs(savepath)

for idx in range(len(files)):
    file = open(savepath + str(idx)+'_image.txt','w')
    snapshot = files[idx]

    img = PIL.Image.open(snapshot)

    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)
    bbox.resize([bbox.size // 11, 11])

    for k, b in enumerate(bbox):
        n = b[0:3]
        theta = np.linalg.norm(n)
        n /= theta
        R = rot(n, theta)
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(8)])
        vert_2D = vert_2D / vert_2D[2, :]

        maxX = max(vert_2D[0,:])
        minX = min(vert_2D[0,:])
        maxY = max(vert_2D[1,:])
        minY = min(vert_2D[1,:])

        if minX < 0:
            minX = 0
        if minY < 0:
            minY = 0
        if maxX >= 1914:
            maxX = 1913
        if maxY >=1052:
            maxY = 1051


        X = (maxX + minX) / 2
        Y = (maxY + minY) / 2
        width = maxX - minX
        height = maxY - minY
        category = b[9]
        box_2d = [category, X, Y, width, height]
        for number in box_2d:
            file.write('%d ' %number)
        file.write('\n')
    file.close()
    img.save(savepath + str(idx) + '_image.jpg')
