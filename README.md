# PyCilium
Python functions to analyze the length of cilia in confocal stacks.

## Rationale
The idea is to provide a comfortable user interface for annotating z projection images and measure the length of cilia. ROI-related information should be saved in text files.

## Prerequisites

```
sudo apt-get install python3-tk
```
Might be a good idea to work in a virtual environment. Create one for PyCilium:
```
$ python3 -m venv pcl
$ source pcl/bin/activate
```
Then in the virtual environment, `pip3 install` the following packages: `opencv-python`, `wheel`, `jupyter`, `python-bioformats`, `pickle-mixin`, `numpy`, `pandas`, `Pillow` and `python-resize-image`.

## Data organizarion

    DATA_CILIA (equivalent to workspace3/POC5_project/osteoblast)
    |-- 20200121 (date)
    |   |
    |   |-- Project1 (or any other projet name)
    |   |   |
    |   |   |-- Project1.lif (lif file saved by Leica software)
    |   |   |-- parsed_metadata.csv
    |   |   |-- parsed_metadata.pickle
    |   |   |-- S01_OB002_P2_B1_X65_FOP-647_Poc5-488_GT335-555 (series folder created by PyCilium)
    |   |   |   |-- z_proj_chan1.png (z projection created by PyCilium)
    |   |   |   |-- z_proj_chan2.png
    |   |   |   |-- z_proj_chan3.png
    |   |   |-- [...] other series
    |   |
    |   |-- Project2
    |       |
    |       |-- Project2.lif
    |       |-- parsed_metadata.csv
    |       |-- parsed_metadata.pickle
    |       |-- [...]
    |
    |-- 20200123 (another date)
        |
        |-- Project3
            |-- [...]

## Nomenclature
The **date** is the image acquisition date. A **project** file is the bundle file saved by the Leica software (extension: .lif). Several projects can be saved for one date. A project is subdivided in **series** which are basically different stacks. One series contains usually several **channels**.

## Steps of the analysis

### Step 1: loading data from lif file, parsing metadata, selecting channels based on projections
This is all done by:
```python
import GUI
root = GUI.Root()  # Use class Root defined in GUI/TkDialog
root.mainloop()  # Run Tk interface
```
root contains useful variables:  
`root.fullpath` = path to lif file  
`root.md` = metadata of lif file  
`root.series_indx` = index of selected series  
`root.series_name` = name of selected series  
`root.selected_chans` = indices of selected channels  
`root.contains_cilia` = index of channel containing cilia

Below: screenshot of the Tkinter interface for selecting the lif file, selecting the series and channels. For each channel, a slide bar enables pixel saturation adjustment and a checkbox is used to indicate which channel contains cilia (on which length measurements will be performed). Only one checkbox can be selected.

<img src="https://github.com/ghyomm/PyCilium/blob/master/pics/tk_GUI.png" width="60%">

This procedures creates a bunch of files (metadata) and folders (one per series, with z projection images inside) in the original directory containing the lif file, see directory before and after:

<img src="https://github.com/ghyomm/PyCilium/blob/master/pics/data_folder.jpg" width="60%">

`root.md` contains the metadata extracted from the lif file (saved as csv and pickle files).

```
	ID	Name	Nchan	SizeX	PhysicalSizeX	PhysicalSizeXUnit	SizeY	PhysicalSizeY	PhysicalSizeYUnit	SizeZ	PhysicalSizeZ	PhysicalSizeZUnit	PixelType
0	Image:0	OB002_P2_B1_X65_FOP-647_Poc5-488_GT335-555	3	1024	0.180375	µm	1024	0.180375	µm	13	0.500203	µm	uint8
1	Image:1	OB002_P2_B1_X40_FOP-647_Poc5-488_GT335-555	3	1024	0.284091	µm	1024	0.284091	µm	10	0.800134	µm	uint8
2	Image:2	OB004_P2_B2_X40_FOP-647_Poc5-488_GT335-555	3	512	0.568738	µm	512	0.568738	µm	13	0.500203	µm	uint8
3	Image:3	OB004_P2_B2_X40_FOP-647_Poc5-488_GT335-555_1	3	1024	0.284553	µm	1024	0.284553	µm	11	0.800133	µm	uint8
4	Image:4	OB004_P2_B2_X63_Z1.68_FOP-647_Poc5-488_GT335-555	3	1024	0.107079	µm	1024	0.107079	µm	11	0.500203	µm	uint8
5	Image:5	OB004_P3_B2_X63_Z1.68_alphatub-647_Poc5-488_Fo...	3	1024	0.107428	µm	1024	0.107428	µm	16	0.199795	µm	uint8
6	Image:6	OB004_P3_B2_X63_Z1.68_alphatub-647_Poc5-488_Fo...	3	1024	0.107602	µm	1024	0.107602	µm	12	0.500203	µm	uint8
7	Image:7	OB004_P3_B2_X63_alphatub-647_Poc5-488_Fop-555	3	1024	0.179793	µm	1024	0.179793	µm	15	0.500203	µm	uint8
```

### Step 2: working with projections images and OpenCV for drawing ROIs interactively.
We now have all the info to start working on z projection images. The code below shows how to obtain the hyperstack of a series (a 4-dimensional stack containing 3D stacks for all channels).

```python
import bioformats as bf
import numpy as np
rdr = bf.ImageReader(root.fullpath)
# Obtain hyperstack
stack = []
for z in range(root.md.SizeZ[root.series_indx]):  # Loop through z slices
    im = rdr.read(z=z, series=root.series_indx, rescale=False)
    stack.append(im)
stack = np.array(stack)
```

The projection image is obtained as follows:
```python
# Compute z projection for channel containing cilia
proj = np.amax(stack[:,:,:,root.contains_cilia],0)
```

The next step is to display the projection image using OpenCV and use mouse events to draw a ROI:

```python
import roi
my_roi = roi.RoiCilium(proj,'Set threshold and draw bounding polygon')  # Initialize class RoiCilium
my_roi.contour.draw_contour()
```


