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

### Loading data from lif file, parsing metadata, selecting channels based on projections
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
