# PyCilium
Python functions to analyze the length of cilia in confocal stacks. Several cilia can be analyzed in the same stack. Length is calculated in 3D.

## Data files organizarion

    DATA_CILIA (equivalent to workspace3/POC5_project/osteoblast)
    |-- 20200121 (date)
    |   |-- Project1
    |       |-- Project1.lif (lif file saved by Leica software)
    |       |-- S01_OB002_P2_B1_X65_FOP-647_Poc5-488_GT335-555
    |       |-- [...] One folder per series (= one stack)
    |       |-- S05_OB004_P2_B2_X63_Z1.68_FOP-647_Poc5-488_GT335-555
    |   |-- Project2
    |       |-- [...]
    |-- 20200123 (other date)
        |-- [...]

## Steps of the analysis

1. Open .lif file (Leica) from folder. If a previous analysis was run, files related to existing ROIs (e.g. ROI01, ROI02, etc) are detected and new ROIs will are saved with proper name (e.g. ROI3, ROI4, etc).
2. The user sets an appropriate threshold around a selected cilium, such that only a few pixels in the cilium are saturated. He/she then draws a bounding polygon around the cilium.
