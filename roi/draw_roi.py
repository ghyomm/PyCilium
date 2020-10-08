from typing import List, Optional
from scipy.spatial.distance import cdist
from uuid import uuid4
import json
import numpy as np
import cv2
import sys
import time
import javabridge
from scipy.interpolate import UnivariateSpline, RectBivariateSpline


class Point:

    def __init__(self, x, y) -> None:
        super().__init__()
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError()

    def __iter__(self):
        cs = (self.x, self.y)
        for c in cs:
            yield  c

    def __repr__(self):
        return f'Point at {self.x, self.y}'

    def draw(self, img, color=(0, 255, 0)):
        cv2.circle(img, (self.x, self.y), 2, color, -1)


class ROI:
    def __init__(self, data_dict: Optional[dict] = None) -> None:
        super().__init__()
        self._pts: List[Optional[Point]] = []
        self._id = uuid4()
        self._closed = False
        self.contour: Optional[np.ndarray] = None
        if data_dict is not None:
            self.from_dict(data_dict)

    @property
    def closed(self):
        return self._closed

    @property
    def id(self):
        return self._id

    @property
    def points(self):
        return self._pts

    def push(self, pt: Point):
        self._pts.append(pt)

    def pop(self, ix: Optional[int] = -1):
        self._pts.pop(ix)

    def add_pt(self, pt: Point):
        ix = self.closest_point(*pt)
        if ix is None:
            self.push(pt)
            return
        if ix == 0 and len(self._pts) > 2:
            self._closed = True
            self.push(pt)
            return
        offset = 1
        pts = self._pts[:ix+offset]
        pts.append(pt)
        pts.extend(self._pts[ix+offset:])
        self._pts = pts

    def draw(self, img, pt_color=(0, 255, 0), l_color=(0, 255, 0)):
        for ix, pt in enumerate(self._pts[:-1]):
            pt.draw(img, pt_color)
            n_pt = self._pts[ix + 1]
            cv2.line(img, (pt.x, pt.y), (n_pt.x, n_pt.y), l_color, 1)
        if len(self._pts) >= 1:
            self._pts[-1].draw(img, pt_color)
        if self.closed:
            cv2.line(img, (self._pts[-1].x, self._pts[-1].y), (self._pts[0].x, self._pts[0].y),
                     l_color, 1)
        if self.contour is not None:
            cv2.drawContours(img, [self.contour], 0, (125, 125, 125), 1)

    def remove_closest(self, x, y):
        ix = self.closest_point(x, y)
        if ix is not None:
            if ix == 0 or ix == -1:
                self._closed = False
            self.pop(ix)

    def is_point_inside(self, x, y):
        inside = cv2.pointPolygonTest(self._pts_array(), (x, y), False)
        return inside > 0

    def closest_point(self, x, y):
        pts_arr = self._pts_array()
        if len(pts_arr.shape) != 2:
            return None
        dist = np.squeeze(cdist(np.array([(x, y)]), pts_arr))
        print((x,y), pts_arr, dist)
        return np.argmin(dist)

    def _pts_array(self):
        a = np.array([(pt.x, pt.y) for pt in self._pts], dtype=np.int32)
        return a

    def to_dict(self):
        d = {'id': str(self.id), 'points': self._pts_array().tolist(),
             'closed': self.closed,
             'contour': self.contour.tolist()}
        return d

    def from_dict(self, data):
        self._id = data['id']
        self._pts = [Point(pt[0], pt[1]) for pt in data['points']]
        self._closed = data['closed']
        self.contour = np.array(data['contour'])

    def get_mask(self, img: np.ndarray):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self._pts_array(), 1)
        return mask


class RoiCilium:
    def __init__(self, imsrc, msg):
        self.contour = DrawCiliumContour(imsrc, msg)


class DrawCiliumContour:
    def __init__(self, im, msg):
        self.im = im  # Source image
        # self.pts = []  # Coordinates of bounding points
        self.rois: List[ROI] = []
        self.c_roi: Optional[ROI] = None
        self.all_rois: Optional[List[dict]] = None
        self.handler = msg
        self.im_copy1 = self.im.copy()  # Original copy
        self.im_copy2 = self.im.copy()  # Original copy
        self.immask = np.zeros_like(im, dtype=np.uint8)  # Prepare mask
        self.c_mask = np.zeros_like(im, dtype=np.uint8)  # Prepare mask
        self.mask_mode = False
        self.th = None
        self.closest = None
        self.closest_last = None

    def update_rois(self):
        self.im_copy1 = self.im_copy2.copy()  # Reinitialize image
        for c_roi in self.rois:
            c_roi.draw(self.im_copy1)

    def draw_contour(self):
        """
        Main function of class DrawCiliumContour
        """
        cv2.namedWindow(self.handler,
                        flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.handler, 1000, 1000)
        cv2.createTrackbar('Threshold', self.handler, 110, 255, self.callback_trackbar)
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html#code-demo
        # cv2.createTrackbar('0 : OFF \n1 : ON', 'image',0,1,nothing)
        cv2.setMouseCallback(self.handler, self.callback_mouse)  # Bind window to callback_mouse
        cv2.imshow(self.handler, self.im_copy1)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Hit `q` to exit
                self.exit()
                break  # Return to main function
            elif key == ord('n'):
                # New ROI
                print('>>> NEW ROI')
                self.c_roi = ROI()
                self.rois.append(self.c_roi)
            elif key == ord('m'):
                self.mask_mode = not self.mask_mode
                if self.mask_mode:
                    self.im_copy1 = self.im_copy2 = cv2.applyColorMap(255*self.immask, cv2.COLORMAP_HOT)
                else:
                    self.im_copy1 = self.im_copy2 = self.im.copy()
            elif key == ord('c') and self.c_roi.closed:
                self.c_mask = self.c_roi.get_mask(self.im_copy1)
                self.segment_cilia()
            elif key == ord('p'):
                print('PAUSE')
            # time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def segment_cilia(self):
        g_im = cv2.GaussianBlur(self.im, (5, 5), 5)
        roi_masked = g_im * self.c_mask
        mask_notroi = g_im * (1 - self.c_mask)
        med = np.median(mask_notroi)
        mad = np.median(np.abs(mask_notroi - med))
        # th = np.quantile(roi_masked[roi_masked > 0], .90)
        th = med + 5*mad
        _, th_cil = cv2.threshold(roi_masked, th, 255, cv2.THRESH_BINARY)
        all_cnt, _ = cv2.findContours(th_cil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Keep the biggest object
        cnt = max(all_cnt, key=lambda x: cv2.contourArea(x))
        self.c_roi.contour = np.squeeze(cnt)
        cv2.imshow('Thresholded', th_cil)

    def exit(self):
        javabridge.detach()
        self.all_rois = [r.to_dict() for r in self.rois]
        cv2.destroyWindow(self.handler)

    def callback_trackbar(self, event):
        '''
        Callback function responding to trackbar
        Updates working copy of image as a function of threshold
        '''
        self.th = cv2.getTrackbarPos('Threshold', self.handler)
        if self.th == 0:
            self.th = 1
        ret, tmp1 = cv2.threshold(self.im, self.th, 255, cv2.THRESH_TRUNC)
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
        cv2.imshow(self.handler, self.im_copy1)

    def callback_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and self.c_roi is not None:
            self.c_roi.add_pt(Point(x, y))
            if self.c_roi.closed:
                mask = self.c_roi.get_mask(self.im_copy1)
                self.immask += mask
        if event == cv2.EVENT_MBUTTONDOWN and self.c_roi is not None:
            self.c_roi.remove_closest(x, y)
            # diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            # del self.pts[diff.argmin()]
            # diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            # self.closest = self.pts[diff.argmin()]
            self.im_copy1 = self.im_copy2.copy()  # Reinitialize image
            # self.closest_last = self.closest
        # if event == cv2.EVENT_MOUSEMOVE and (len(self.pts) > 0):
        #     Find reference point closest to latest mouse position (= closest reference point)
        # diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
        # self.closest = self.pts[diff.argmin()]
        # Update coordinates of closest reference point
        # if self.closest_last != self.closest:
        #     self.im_copy1 = self.im_copy2.copy()
        #     self.closest_last = self.closest
        # self.draw_pts_lines()
        self.update_rois()
        cv2.imshow(self.handler, self.im_copy1)


def fit_cilium(im: np.ndarray, th_cil: np.ndarray, cnt: np.ndarray):
    """
    Fit a segmented cilium to extract it and its ridge for statistics

    Parameters
    ----------
    im: np.ndarray
    th_cil: np.ndarray
    cnt: np.ndarray

    Return
    ------
    cilium: dict
    ridge: dict

    Example
    -------
    # To check the results
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.gcf()
    >>> ax = fig.add_subplot(1, 1, 1, projection='3d')
    >>> ax.plot_trisurf(**cilium, antialiased=True, cmap=plt.cm.Spectral)
    >>> ax.plot(ridge['x'], ridge['y'], ridge['z'] + 5, lw=3)
    """
    gi = th_cil > 0
    x, y = np.where(gi)
    z = im[gi]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    n = len(z.reshape(-1))
    bs = RectBivariateSpline(np.arange(xmin, xmax), np.arange(ymin, ymax), z, s=n*5)
    ze = bs.ev(x, y)
    fc = np.zeros_like(im)
    fc[x, y] = ze
    ux = np.unique(x)
    uy = np.unique(y)
    by = np.argmax(fc, 1)[ux]
    bx = np.argmax(fc, 0)[uy]
    sp = UnivariateSpline(ux, by, s=5*len(ux))
    sp2 = UnivariateSpline(uy, bx, s=5*len(ux))
    f_x = sp2(by)
    f_y = sp(bx)
    if len(ux) > len(uy):
        order = np.argsort(bx)
        xs = bx[order]
        ys = f_y[order]
    else:
        order = np.argsort(by)
        xs = f_x[order]
        ys = by[order]
    ridge_z = bs.ev(xs, ys)
    cil_len = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
    cilium = {'x': x, 'y': y, 'z': ze, 'length': cil_len}
    ridge = {'x': xs, 'y': ys, 'z': ridge_z}

    return cilium, ridge


if __name__ == '__main__':
    import bioformats as bf
    import numpy as np
    import roi

    # Initialize javabridge etc
    bf.javabridge.start_vm(class_path=bf.JARS, run_headless=True)
    log4j = javabridge.JClassWrapper("loci.common.Log4jTools")
    log4j.setRootLevel("OFF")  # Turn off debug messages

    ROOT_PATH = '/home/remi/Programming/TDS/InProgress/PyCilium/20201006/Project1/Project1.lif'

    rdr = bf.ImageReader(ROOT_PATH)
    # Obtain hyperstack
    stack = []
    for z in range(16):  # Loop through z slices
        im = rdr.read(z=z, series=5, rescale=False)
        stack.append(im)
    stack = np.array(stack)
    # Compute z projection for channel containing cilia
    proj = np.amax(stack[:, :, :, 2], 0)
    my_roi = roi.RoiCilium(proj, 'Set threshold and draw bounding polygon')
    my_roi.contour.draw_contour()

