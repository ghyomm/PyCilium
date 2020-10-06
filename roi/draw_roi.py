# import read_lif
import numpy as np
import cv2
import time


class RoiCilium:
    def __init__(self, imsrc, msg):
        self.contour = DrawCiliumContour(imsrc, msg)


class DrawCiliumContour:
    def __init__(self, im, msg):
        self.im = im  # Source image
        self.pts = []  # Coordinates of bounding points
        self.handler = msg
        self.im_copy1 = self.im.copy()  # Original copy
        self.im_copy2 = self.im.copy()  # Original copy
        self.immask = np.zeros_like(im, dtype=np.uint8)  # Prepare mask
        self.th = None
        self.closest = None
        self.closest_last = None

    def draw_pts_lines(self):
        '''Add points and lines drawn by user'''
        if(len(self.pts)>0):
            for i in range(len(self.pts)):  # Draw points
                cv2.circle(self.im_copy1, tuple(self.pts[i]), 1, (0, 255, 0), -1)
            for i in range(len(self.pts) - 1):  # Draw lines
                cv2.line(self.im_copy1, tuple(self.pts[i]), tuple(self.pts[i + 1]), (0, 255, 0), 1)
            if self.closest is not None:
                cv2.circle(self.im_copy1, tuple(self.closest), 3, (255, 255, 0), 1)

    def draw_contour(self):
        '''
        Main function of class DrawCiliumContour
        '''
        cv2.namedWindow(self.handler)
        cv2.createTrackbar('Threshold', self.handler, 255, 255, self.callback_trackbar)
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html#code-demo
        cv2.createTrackbar('0 : OFF \n1 : ON', 'image',0,1,nothing)
        cv2.setMouseCallback(self.handler, self.callback_mouse)  # Bind window to callback_mouse
        cv2.imshow(self.handler,self.im_copy1)
        while True:
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                if (len(self.pts)>2):
                    cv2.fillPoly(self.immask, np.array([self.pts]), (255, 255, 255), 1)
                cv2.destroyWindow(self.handler)
                break  # Return to main function
            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def callback_trackbar(self, event):
        '''
        Callback function responding to trackbar
        Updates working copy of image as a function of threshold
        '''
        self.th = cv2.getTrackbarPos('Threshold', self.handler)
        ret, tmp1 = cv2.threshold(self.im, self.th, 255, cv2.THRESH_TRUNC)
        if(self.th == 0):
            self.th = 1
        self.im_copy1 = (255 * (tmp1.astype('float32') / self.th)).astype('uint8')
        ret, tmp2 = cv2.threshold(self.im_copy1, 254, 255, cv2.THRESH_BINARY)
        sat = (tmp2 / 255).astype('uint8')  # Saturated pixels = 1
        gi = np.where(sat==1)  # Identify saturated pixels
        self.im_copy1 = cv2.applyColorMap(self.im_copy1, cv2.COLORMAP_HOT)
        sat_col = [255, 0 ,0]  # Saturated pixels in blue
        for j in range(3):
            for i in range(len(gi[0])):
                self.im_copy1[gi[0][i],gi[1][i],j] = sat_col[j]
        self.im_copy2 = self.im_copy1.copy()  # Keep copy without pts and lines
        cv2.imshow(self.handler,self.im_copy1)

    def callback_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([x, y])
        if event == cv2.EVENT_MBUTTONDOWN and (len(self.pts) > 0):
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            del self.pts[diff.argmin()]
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            self.im_copy1 = self.im_copy2.copy()  # Reinitialize image
            self.closest_last = self.closest
        if event == cv2.EVENT_MOUSEMOVE and (len(self.pts) > 0):
            # Find reference point closest to latest mouse position (= closest reference point)
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            # Update coordinates of closest reference point
            if self.closest_last != self.closest:
                self.im_copy1 = self.im_copy2.copy()
                self.closest_last = self.closest
        self.draw_pts_lines()
        cv2.imshow(self.handler,self.im_copy1)
