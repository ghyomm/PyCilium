

import cv2, os, math, operator, sys, time
import numpy as np


def imRescale2uint8(im):
    '''Rescale and convert to 8 bit ([min max] to [0 255])'''
    imOffset = im - np.min(im)
    imOut = 255 * (imOffset / np.max(imOffset))
    return(imOut.astype('uint8'))


def imLevels(im,low,high):
    '''Apply 2 thresholds: values < low are set to low, values > high are set to high'''
    ilow = np.where(im <= low) # Indices of pixels < low
    ihigh = np.where(im >= high)
    for i in list(zip(ilow[0],ilow[1])):
        im[i]=low
    for i in list(zip(ihigh[0], ihigh[1])):
        im[i]=high
    return(im)

def imLevelHigh(im,high):
    '''Scale (0,high) pixel values to (0,255)'''
    if high > 0 & high <= 255:
        im_adjusted = np.floor(255 * (im / high)).astype('int')
        ihigh = np.where(im_adjusted > 255)
        for i in list(zip(ihigh[0], ihigh[1])):
            im_adjusted[i]=255
        im_adjusted = im_adjusted.astype('uint8')
    else:
        sys.exit('Argument high needs to be an integer > 0 and <= 255.')
    return(im_adjusted)

def customResize(im,factor):
    '''Resize image using non integer factor'''
    imResized=cv2.resize(im, tuple(map(math.floor, tuple(factor * x for x in im.shape[0:2]))), interpolation=cv2.INTER_AREA)
    truefactor=tuple(map(operator.truediv, imResized.shape, im.shape))[0]
    return imResized, truefactor

def myimshow(im):
    cv2.imshow('im',im)
    while True:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            cv2.destroyWindow('im')
            break
        time.sleep(0.01)  # Slow down while loop to reduce CPU usage
