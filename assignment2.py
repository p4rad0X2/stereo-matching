#///CS4186 ASSIGNMENT 2
#///PRATUL RAJAGOPALAN
#///55858290


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy
import math

def func(inpath, outpath):
    path1 = inpath + '/view1.png'
    path2 = inpath + '/view5.png'
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)


    sto = cv2.StereoSGBM_create(blockSize=1,minDisparity=0,P1=1, P2=80, numDisparities=256, uniquenessRatio=0, disp12MaxDiff=100, speckleWindowSize = 0)
    disp= sto.compute(img1, img2)

    disp = cv2.normalize(disp, disp, alpha=248, beta=12, norm_type=cv2.NORM_MINMAX)
    disp = np.uint8(disp)
    blur = cv2.GaussianBlur(disp, (111,111), 0)
    blur2 = cv2.bilateralFilter(blur, 1, 111, 111)
    out = outpath + '\disp1.png'
    cv2.imwrite(out, blur2)





func("./Art", r"Z:\CS4186\Assignment 2\pred\Art")

func("./Dolls", r"Z:\CS4186\Assignment 2\pred\Dolls")

func("./Reindeer", r"Z:\CS4186\Assignment 2\pred\Reindeer")


def psnr(img1, img2):
    mse = numpy.mean( ((img1 - img2)) ** 2 )
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test():
    test_imgs = ["Art", "Dolls", "Reindeer"]
    
    for index in range(3):
        gt_names = r"C:/Users/pratu/Dropbox/My PC (LAPTOP-JLO2E8PQ)/Desktop/PSNR_Assignment2/PSNR_Python/gt/"+test_imgs[index]+"/disp1.png"
        gt_img = numpy.array(Image.open(gt_names),dtype=float);
    
        
        pred_names =  "./pred/"+test_imgs[index]+"/disp1.png";
        pred_img = numpy.array(Image.open(pred_names),dtype=float);
        
# When calculate the PSNR:
# 1.) The pixels in ground-truth disparity map with '0' value will be neglected.
# 2.) The left part region (1-250 columns) of view1 is not included as there is no
#   corresponding pixels in the view5.
        [h,l] = gt_img.shape
        gt_img = gt_img[:, 250:l]
        pred_img = pred_img[:, 250:l]
        pred_img[gt_img==0]= 0
    
        peaksnr = psnr(pred_img,gt_img);
        print('The Peak-SNR value is %0.4f \n', peaksnr);



if __name__== '__main__':
    test()  



