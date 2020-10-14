# Notes on PyCillium

## Installation



```bash
mkvirtualenv pcl
sudo apt-get install openjdk-11-jdk
pip install numpy pandas Pillow wheel jupyter opencv-python python-bioformats python-resize-image pickle-mixin
% Optional
pip install matplotlib
% New
pip install scipy
```

## Corrections

+ Paths hard-coded in `GUI/TkDialog` replaced with some `pathlib` tricks which need to be checked more carefully
  + There is a strong requirement for the folder structure: `YYYMMDD/ProjectName/Stack.lif`

## Startup code

```python
import GUI
root = GUI.Root()  # Use class Root defined in GUI/TkDialog
root.mainloop()  # Run Tk interface
```

Select third image from the bottom of the list, check the last channel 'contains cilia' checkbox and validate

```python
import bioformats as bf
import numpy as np
import matplotlib.pyplot as plt
rdr = bf.ImageReader(root.fullpath)
# Obtain hyperstack
stack = []
for z in range(root.md.SizeZ[root.series_indx]):  # Loop through z slices
    im = rdr.read(z=z, series=root.series_indx, rescale=False)
    stack.append(im)
stack = np.array(stack)
# Compute z projection for channel containing cilia
proj = np.amax(stack[:,:,:,root.contains_cilia],0)
plt.ion()
plt.imshow(proj, vmax=50)
```

```python
import roi
my_roi = roi.RoiCilium(proj,'Set threshold and draw bounding polygon')  # Initialize class RoiCilium
my_roi.contour.draw_contour()
```

## Comments and bug tracking Oct 14

### Major:
  - In manual ridge drawing mode : not possible to erase drawing or remove a point (points are not drawn anyway). Middle click is still operating on ROI contour in manual ridge drawing mode.
  - After having finished to draw a ROI contour (but 'e' has not been pressed), pressing 'm' does not initiate manual mode right away. Looks like 'm' has to be pressed a secdon time.

### Minor:
  - ROI number is displayed next to first point, the user must be aware of this otherwise it may superimpose on the cilium.
  - Left click inside an existing ROI does not seem to toggle the ROI drawing mode. Control of the ROI drawing mode comes back after some frenetic clicking, not sure how this happens.
  - Ridge line would be better in flashy green.
  - Would it be a lot of work to add an option (new key stroke) to erase MAD threshold line and ridge line but not the ROI contour ? In case the user does not like what the automatic ridge calculation does and wants to use manual mode (and alternatively erasing a manually drawn ridge line) --> yes the lazy user could just erase the whole ROI and re-draw it.



