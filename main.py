
'''
First tried Using read_lif to access .lif files: https://pypi.org/project/read-lif/
But realized that the metadata obtained with read_lif is crap (wrong X,Y,Z pixel size)
Then tried bioformats: https://pypi.org/project/python-bioformats/
Found the following post very helpful:
https://ilovesymposia.com/2014/08/10/read-microscopy-images-to-numpy-arrays-with-python-bioformats/
Bioformats gives same pixelsize values as when loaded with Fiji...
...this is reassuring.
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

import read_lif, cv2, time, roi, os, sys
import numpy as np
import pandas as pd
from datetime import date
import utilities as utils  # Custom functions
# from __future__ import absolute_import, print_function, unicode_literals

import javabridge
import bioformats as bf
from bioformats import log4j
from xml.etree import ElementTree as ET

javabridge.start_vm(class_path=bf.JARS,
                    run_headless=True)


class LifFile:
    '''Custom class to handle data path and file name'''
    def __init__(self):
        self.dpath = ''  # Main data path
        self.date = ''  # Project date
        self.name = ''  # Project name


def proj_and_rescale(h, n, rf):
    '''Old function working with read_lif'''
    # h: hyperstack (X x Y x 3 numpy array) produced by getSeries()
    # n: channel number
    # rf: rescale factor
    stk = h.getFrame(T = 0, channel = n)
    proj = np.amax(stk,0)
    im_out = utils.imRescale2uint8(utils.imLevels(proj, proj.min(), 0.2*proj.max()))
    im_out_resized, fac = utils.customResize(im_out, rf)
    cv2.putText(im_out_resized, 'ch. ' + str(n), (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, lineType = cv2.LINE_AA)
    return(im_out_resized)


def proj_channels(lif_file):
    # lif_file: full path to lif file
    reader = read_lif.Reader(lif_file)
    series = reader.chooseSerieIndex() # Let user chose the series
    hyperstack = reader.getSeries()[series]
    im_list = [proj_and_rescale(hyperstack, x, 0.5) for x in [0, 1, 2]]
    im_out = np.concatenate((im_list[0],im_list[1],im_list[2]),axis=1)
    handler = 'Z max projections'
    cv2.namedWindow(handler)
    cv2.imshow(handler,im_out)
    while True:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            cv2.destroyWindow(handler)
            break
        time.sleep(0.01)


def get_pix_size(mdroot,k,attr='PhysicalSizeX'):
    '''Retrieve pixel size from metadata'''
    arr = [any(['PhysicalSize' in i for i in list(m.attrib.keys())]) for m in mdroot[k]]
    ind = int(np.where(arr)[0])  # Index of dictinnary position
    return mdroot[k][ind].attrib.get(attr)


if __name__ == '__main__':
    #
    # Some path definitions
    lif = LifFile()
    lif.path = '/home/ghyomm/DATA_CILIA'
    lif.date = '20200121'
    lif.name = 'Project1.lif'
    project_name = os.path.splitext(lif.name)[0]
    path_to_lif = os.path.join(lif.path,lif.date,project_name)
    #
    # Using bioformats to load and parse metadata
    log4j.basic_config()
    md = bf.get_omexml_metadata(os.path.join(path_to_lif, lif.name))
    mdroot = ET.fromstring(md)
    # Check where to look for information on acquired series in metadata
    vals = [any(['Image' in i for i in list(m.attrib.values())]) for m in mdroot]
    gi = np.where(vals)[0]  # Series name (containing 'Image' string)
    # Generate data frame with series name and pixels sizes
    params = [[mdroot[x].attrib.get('Name') for x in gi],
        [get_pix_size(mdroot,k,attr='PhysicalSizeX') for k in gi],
        [get_pix_size(mdroot,k,attr='PhysicalSizeY') for k in gi],
        [get_pix_size(mdroot,k,attr='PhysicalSizeZ') for k in gi]]
    df = pd.DataFrame(params).transpose()
    df.columns = ['Name', 'SizeX', 'SizeY', 'SizeZ']
    #
    '''CONTINUE FROM HERE : maybe try to continue with bioformats to
    load images rather than with read_lif
    '''

    '''Below: old code running with read_lif'''
    # Check channel z projections to chose channel containing cilia
    proj_channels(os.path.join(path_to_lif, lif.name))
    # Read data from lif file
    reader = read_lif.Reader(os.path.join(path_to_lif,lif.name))
    nseries = len(reader.getSeries())
    print('Found', str(nseries), 'series in', lif.name, 'with names:\n')
    # Create one folder per series, if necessary
    series_names = []
    for i in range(nseries):
        series_names.append(reader.getSeriesHeaders()[i].getName())
        print(series_names[i], '\n')
        series_names[i] = 'S{:0>2d}'.format(i+1) + '_' + series_names[i]
        # Useful ref: https://mkaz.blog/code/python-string-format-cookbook/
        series_folder = os.path.join(path_to_lif,series_names[i])
        if not os.path.exists(series_folder): os.makedirs(series_folder)
    # Analyze one series
    selected_series = reader.chooseSerieIndex() # Let user chose the series
    hyperstack = reader.getSeries()[selected_series]
    # cv2.imwrite(os.path.join(path_to_lif,series_names[selected_series],'im.png'),np.concatenate((proj1,proj2),axis=1))
    # stack.getName()  # Name of stack
    # stack.get2DShape()  # Number of pixels
    stack = hyperstack.getFrame(T = 0, channel = 2)
    proj = utils.imLevels(np.amax(stack,0),0,100)
    my_roi = roi.RoiCilium(proj,'Set threshold and draw bounding polygon')  # Initialize class RoiCilium
    my_roi.contour.draw_contour()
    #
    # Grab mask using contours defined by user
    mask = np.broadcast_to(my_roi.contour.immask == 0, stack.shape)
    # Apply mask to original stack
    stack_masked = np.ma.array(stack, mask=mask, fill_value=0).filled()
    # Grab points of contour defined by user
    pts = np.array(my_roi.contour.pts)
    # Define limits of contour bounding box
    left = min(pts[:,1])
    right = max(pts[:,1])
    top = min(pts[:,0])
    bottom = max(pts[:,0])
    # Keep mini stack (region of stack inside bounding box)
    mini_stack = stack_masked[:,left:right,top:bottom]
    # Save projections of mini_stack
    proj_top = utils.imRescale2uint8(np.amax(mini_stack,0))

    fac = 5 # scale factor
    dim = (int(proj_top.shape[1] * fac), int(proj_top.shape[0] * fac))
    proj_top_resized = cv2.resize(proj_top, dim, interpolation = cv2.INTER_AREA)
    # mask = np.broadcast_to(my_roi.contour.immask == 0, stack.shape)
    # stack_cilium = np.ma.array(stack, mask=mask, fill_value=0).filled()
    # proj_top = np.amax(stack_cilium,1)
    # im = utils.imRescale2uint8(proj_top)
    cv2.imshow('im',proj_top_resized)
    while True:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            cv2.destroyWindow('im')
            break
            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

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
