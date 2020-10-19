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

Below: screenshot of the Tkinter interface for selecting the lif file, selecting the series and channels. For each channel, a slide bar enables pixel saturation adjustment and a checkbox is used to indicate which channel contains cilia (on which length measurements will be performed). Only one checkbox can be selected. In the example below, one cilium is visible in the projection of channel 3.

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

We now have all the info to start working on z projection images. 
The hyperstack of a series (a 4-dimensional stack containing 3D stacks for all channels) is reduced to maximum projections for each channel.
The projection with cilia, labeled in the first step, will be used to draw ROIs during this step.
The OpenCV interface to do so can be started by pressing the 'OK' button in the TKinter window

#### Usage:
+ Use the bottom trackbar to adjust contrast (*threshold*) for better visualization of cilia. This does not affect analysis.
+ Press `n` to start drawing a ROI using the mouse **left** click
+ Use mouse **middle** click to remove the point closest to the mouse pointer
+ Press `d` while the mouse pointer is **inside** a ROI to delete it
+ Press `e` to get out of the ROI drawing mode.
+ *Left* click inside an ROI to enter ROI drawing mode, for this ROI
+ Press `c` to start automatic analysis of the ROI (see below for details)
+ One can adjust the threshold used for cilium detection by moving the *k-MAD* slider at the bottom
+ Press `m` to enter manual ridge line mode. The left mouse click can now be used to draw a line at the center of the cilium
+ Add extra channels (the max-projections from other channels) by pressing `Ctrl-P` and then selecting which channel to add and adjusting its threshold.
+ Save a screenshot of the current display by pressing `Ctrl-S`
+ Press `q` to quit

ROI number is automatically displayed next to the first point

#### Extra details

When quitting, the ROIs and their analysis results are saved into a JSON file in the same folder 
as the original `.lif` file. 
When loading the same `lif` file again, the previous analysis is restored.
 
The JSON structure is the following:
```json
[
  {
    "id": "str, UUID", 
    "points": [["roi_x0", "roi_y0"],["roi_x1", "roi_y1"]],
    "closed": true,
    "contour": [["x0", "y0"],["x1", "y1"]],
    "k_mad": 5,
    "cilium":{"x": [1, 2, 3], "y":  [1, 2, 3], "z":  [1, 2, 3]},
    "ridge": {"x": [1, 2, 3], "y":  [1, 2, 3], "z":  [1, 2, 3], "length":  12.5}
  }
]
```
+ `z` means fluorescence level (either raw or spline estimated)
+ `cilium` contains information about the threshold structure, considered as the cilium
+ `ridge` contains information about the extracted ridge, which implies some smoothing and contains a length (in pixels) estimation

The cilium is extracted using the `fit_cilium` function from `draw_roi.py`. 
Briefly, the raw image is thresholded using median + k * MAD with k being user-adjusted.
A rectangular bivariate spline, with smoothing, is fitted to the corresponding fluorescence signal over the rectangular bounding box of the non-zero pixels.
This spline is then used to extract the ridge line (maxima). 
An other spline with smoothing is fitted to those maxima, and fluorescence signal is estimated. This will be saved along side the maxima coordinates.
Length of the ridge line is evaluated by approximating its integral using the traditional rectangle method.

#### Possible improvements: 
+ Cleaning the MAD-thresholded image (median filter, keeping only the biggest element after contour finding...)
+ Adjustable smoothing level for the different splines.
+ Better placement of ROI number
+ Coloring of ROIs is not correct when reloading from file

The image below shows a cilium with the saturation level adjusted by the user (using a slide bar at the bottom of the window, not visible here; blue pixels are saturated). 
The green polygon was drawn by the user (left click to draw a point).


<img src="https://github.com/ghyomm/PyCilium/blob/master/pics/cilium.png" width="30%">
