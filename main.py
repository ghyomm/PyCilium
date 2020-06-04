#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
First tried Using read_lif to access .lif files: https://pypi.org/project/read-lif/
But realized that the metadata obtained with read_lif is crap (wrong X,Y,Z pixel size)
Then tried bioformats: https://pypi.org/project/python-bioformats/
Bioformats gives same pixelsize values as when loaded with Fiji, which is reassuring.
'''

'''
Data file organization:
DATA_CILIA (equivalent to workspace3/POC5_project/osteoblast)
|-- 20200121 (date)
|   |-- Project1
|       |-- Project.lif (lif file saved by Leica software)
|       |-- S01_OB002_P2_B1_X65_FOP-647_Poc5-488_GT335-555
|       |-- [...] One folder per series (stack)
|       |-- S05_OB004_P2_B2_X63_Z1.68_FOP-647_Poc5-488_GT335-555
|   |-- Project2
|       |-- [...]
|-- 20200123 (other date)
    |-- [...]
'''

import os, sys, lif, re
import numpy as np
import utilities as utils
from tkinter import filedialog
from tkinter import *

# import cv2, time, roi, os, sys, pickle
# import numpy as np
# import pandas as pd
# from datetime import date
# import utilities as utils  # Custom functions

if __name__ == '__main__':
    root = Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename =  filedialog.askopenfilename(initialdir = "/home/ghyomm/DATA_CILIA",
        title = "Select .lif file",filetypes = (("lif files","*.lif"),("all files","*.*")))
    path_comp = utils.splitall(filename)  # Get all components of path
    p = re.compile('^20[0-9]{6}$')  # regex for date format yyyymmdd
    res = np.where([bool(p.match(x)) for x in path_comp])[0]  # Which path component matches regex
    if(len(res)==1):
        date = path_comp[int(res)]  # Get date from folder name
    else:
        sys.exit('Several folders in path match date format yyyymmdd.')
    lif = lif.LifFile('/home/ghyomm/DATA_CILIA',date,path_comp[-1])
    lif.get_metadata(save=True)  # Puts metadata in lif.md
    lif.get_proj()

    # # Analyze one series
    # selected_series = reader.chooseSerieIndex() # Let user chose the series
    # hyperstack = reader.getSeries()[selected_series]
    # # cv2.imwrite(os.path.join(path_to_lif,series_names[selected_series],'im.png'),np.concatenate((proj1,proj2),axis=1))
    # # stack.getName()  # Name of stack
    # # stack.get2DShape()  # Number of pixels
    # stack = hyperstack.getFrame(T = 0, channel = 2)
    # proj = utils.imLevels(np.amax(stack,0),0,100)
    # my_roi = roi.RoiCilium(proj,'Set threshold and draw bounding polygon')  # Initialize class RoiCilium
    # my_roi.contour.draw_contour()
    # #
    # # Grab mask using contours defined by user
    # mask = np.broadcast_to(my_roi.contour.immask == 0, stack.shape)
    # # Apply mask to original stack
    # stack_masked = np.ma.array(stack, mask=mask, fill_value=0).filled()
    # # Grab points of contour defined by user
    # pts = np.array(my_roi.contour.pts)
    # # Define limits of contour bounding box
    # left = min(pts[:,1])
    # right = max(pts[:,1])
    # top = min(pts[:,0])
    # bottom = max(pts[:,0])
    # # Keep mini stack (region of stack inside bounding box)
    # mini_stack = stack_masked[:,left:right,top:bottom]
    # # Save projections of mini_stack
    # proj_top = utils.imRescale2uint8(np.amax(mini_stack,0))
    #
    # fac = 5 # scale factor
    # dim = (int(proj_top.shape[1] * fac), int(proj_top.shape[0] * fac))
    # proj_top_resized = cv2.resize(proj_top, dim, interpolation = cv2.INTER_AREA)
    # # mask = np.broadcast_to(my_roi.contour.immask == 0, stack.shape)
    # # stack_cilium = np.ma.array(stack, mask=mask, fill_value=0).filled()
    # # proj_top = np.amax(stack_cilium,1)
    # # im = utils.imRescale2uint8(proj_top)
    # cv2.imshow('im',proj_top_resized)
    # while True:
    #     if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
    #         cv2.destroyWindow('im')
    #         break
    #         time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    # Save relevant params
    # today = date.today()
    # with open(os.path.join(analysis_path, 'params.txt'), 'w') as outfile:
    #     outfile.write(os.path.join(datapath, struct, session, mouse) + "\n")
    #     outfile.write("Date = " + today.strftime("%Y-%m-%d") + "\n")
    #     outfile.write("Angle = " + str(roi1.horiz.angle) + " deg\n")
    #     outfile.write("Voxel size = " + str(tuple([int(round(1000 * x)) for x in voxel_spacing])) + " um\n")
    #     outfile.write("roi2.top.resize_factor = " + str(roi2.top.resize_factor) + "\n")
    #     outfile.write("roi3.side.resize_factor = " + str(roi3.side.resize_factor) + "\n")
    #     outfile.write("Number of vertebrae annotated = " + str(N_V_annotated) + "\n")
