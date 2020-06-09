

import cv2, os, sys, pickle
import numpy as np
import pandas as pd
import javabridge
import bioformats as bf
import utilities as utils
from bioformats import log4j

# Initialize javabridge etc
bf.javabridge.start_vm(class_path=bf.JARS,run_headless=True)
log4j = javabridge.JClassWrapper("loci.common.Log4jTools")
log4j.setRootLevel("OFF")  # Turn off debug messages
# Think about using javabridge.kill_vm() when program if closed

class LifFile:
    '''Custom class to handle data path and file name'''
    def __init__(self,dpath,date,name):
        self.dpath = dpath  # Main data path
        self.date = date  # Project date
        self.name = name  # File (i.e. project) name with extension
        self.project = os.path.splitext(self.name)[0]
        self.fullpath = os.path.join(self.dpath,self.date,self.project,self.name)
        self.md = pd.DataFrame()  # To store metadata

    def get_metadata(self,save=True):
        # Retrieve metadata info, put it in data frame and save it.
        print('Getting and saving metadata for ' + self.fullpath + '...')
        ome = bf.OMEXML(bf.get_omexml_metadata(self.fullpath))
        md = []
        for ind in range(ome.image_count):
            iome = ome.image(ind)
            md.append([iome.get_ID(),iome.get_Name(),iome.Pixels.get_SizeC(),
                iome.Pixels.get_SizeX(),iome.Pixels.get_PhysicalSizeX(),iome.Pixels.get_PhysicalSizeXUnit(),
                iome.Pixels.get_SizeY(),iome.Pixels.get_PhysicalSizeY(),iome.Pixels.get_PhysicalSizeYUnit(),
                iome.Pixels.get_SizeZ(),iome.Pixels.get_PhysicalSizeZ(),iome.Pixels.get_PhysicalSizeZUnit(),
                iome.Pixels.get_PixelType()])
        self.md = pd.DataFrame(md)
        self.md.columns = ['ID','Name','Nchan',
            'SizeX','PhysicalSizeX','PhysicalSizeXUnit',
            'SizeY','PhysicalSizeY','PhysicalSizeYUnit',
            'SizeZ','PhysicalSizeZ','PhysicalSizeZUnit','PixelType']
        if(save):
            print('Saving metadata...')
            fname = os.path.join(self.dpath,self.date,self.project,'parsed_metadata')
            with open(fname + '.pickle', 'wb') as f:
                pickle.dump(self.md, f)
            self.md.to_csv(fname + '.csv', index=False)
        # Create one folder per series
        project_path = os.path.join(self.dpath,self.date,self.project)
        for i in range(ome.image_count):
            folder_name = 'S{:0>2d}'.format(i+1) + '_' + self.md.Name[i]
            # Useful ref: https://mkaz.blog/code/python-string-format-cookbook/
            series_folder = os.path.join(project_path,folder_name)
            if not os.path.exists(series_folder):
                print('Creating subfolder ' + series_folder + '...')
                os.makedirs(series_folder)

    def get_proj(self):
        # Compute z projections (max intensity) for all channels
        if (self.md.empty):
            sys.exit('get_proj() error: use get_metadata() first.')
        else:
            project_path = os.path.join(self.dpath,self.date,self.project)
            rdr = bf.ImageReader(self.fullpath)
            for i in range(self.md.shape[0]):  # Loop through series
                series_folder = 'S{:0>2d}'.format(i+1) + '_' + self.md.Name[i]
                stack = []
                for z in range(self.md.SizeZ[i]):  # Loop through z slices
                    im = rdr.read(z=z, series=i, rescale=False)
                    stack.append(im)
                stack = np.array(stack)
                # proj_list = []
                # Save one z projection image per channel (if necessary)
                for c in range(self.md.Nchan[i]):
                    proj_file = os.path.join(project_path,series_folder,'z_proj_chan' + str(c+1) + '.png')
                    if(not os.path.exists(proj_file)):
                        print('Saving channel ' + str(c+1) + ' z projection for ' + self.md.Name[i] + '...')
                        cv2.imwrite(proj_file,np.amax(stack[:,:,:,c],0))
                    # below: old code for tiling 3 images in one
                #     prof_im = utils.imRescale2uint8(utils.imLevels(proj,0,150))
                #     cv2.putText(prof_im, 'ch. ' + str(c+1), (20,50),
                #         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, lineType = cv2.LINE_AA)
                #     proj_list.append(prof_im)
                # im_out = np.concatenate((proj_list[0],proj_list[1],proj_list[2]),axis=1)
                # cv2.imwrite(proj_file,im_out)
