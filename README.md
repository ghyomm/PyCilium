# PyCilium
Python functions to analyze the length of cilia in confocal stacks.

## Rationale
The idea is to provide a comfortable user interface for annotating z projection images and measure the length of cilia. ROI-related information should be saved in text files.

## Prerequisites

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

## Steps of the analysis

1. Open .lif file (Leica) from folder. If a previous analysis was run, files related to existing ROIs (e.g. ROI01, ROI02, etc) are detected and new ROIs will are saved with proper name (e.g. ROI3, ROI4, etc).
2. The user sets an appropriate threshold around a selected cilium, such that only a few pixels in the cilium are saturated. He/she then draws a bounding polygon around the cilium.
