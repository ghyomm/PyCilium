#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
First tried Using read_lif to access .lif files: https://pypi.org/project/read-lif/
But realized that the metadata obtained with read_lif is crap (wrong X,Y,Z pixel size)
Then tried bioformats: https://pypi.org/project/python-bioformats/
Bioformats gives same pixelsize values as when loaded with Fiji, which is reassuring.
'''
import sys


if __name__ == '__main__':

    import GUI
    root = GUI.Root()  # Use class Root defined in GUI/TkDialog
    root.protocol("WM_DELETE_WINDOW", root.exit)
    root.mainloop()  # Run Tk interface
    '''
    Below useful variables in root:
    root.fullpath = path to lif file
    root.md = metadata of lif file
    root.series_indx = index of selected series
    root.series_name = name of selected series
    root.selected_chans = indices of selected channels
    root.contains_cilia = index of channel containing cilia
    '''

    # OLD CODE (KEPT JUST IN CASE)
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
