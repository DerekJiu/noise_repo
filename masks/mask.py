
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import pathlib
import re

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def getDenseMask(RL, LL, H, RCLA = None, LCLA = None, imagesize = 1024):
    img = np.zeros([1024,1024])
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 2, -1)
    
    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [H], -1, 3, -1)

    if RCLA is not None:
        RCLA = RCLA.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [RCLA], -1, 4, -1)
    
    if LCLA is not None:      
        LCLA = LCLA.reshape(-1, 1, 2).astype('int')  
        img = cv2.drawContours(img, [LCLA], -1, 5, -1)

    return img



def main():
    for img1 in all_files:
            RL_path = img1.replace('Images','landmarks/RL').replace('.png','.npy')
            LL_path = img1.replace('Images','landmarks/LL').replace('.png','.npy')
            H_path = img1.replace('Images','landmarks/H').replace('.png','.npy')
        
        
            RL = np.load(RL_path)
            LL = np.load(LL_path)
            H = np.load(H_path)
            mask_path = img1.replace('/workspace/Landmark/Images', '/workspace/Landmark/masks').replace('.png', '_mask.png')
            dense_mask = getDenseMask(RL,LL,H)
        
            dense_mask = (dense_mask * 255).astype(np.uint8)
            
            cv2.imwrite(mask_path, dense_mask)
            plt.figure()
            plt.imshow(dense_mask, cmap='gray')
            plt.axis('off')
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == '__main__':
    data_root = pathlib.Path("/workspace/Landmark/Images")
    all_files = list(data_root.glob('*.png'))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)

    main()