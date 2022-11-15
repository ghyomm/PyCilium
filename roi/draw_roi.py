from pathlib import Path
from typing import List, Optional
from scipy.spatial.distance import cdist
from uuid import uuid4
import json
import numpy as np
import cv2
import javabridge
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from functools import partial


SAT_COLOR = (255, 155, 25)
CONTOUR_COLOR = (125, 125, 125)
RIDGE_PT_COLOR = (100, 50, 150)
RIDGE_LINE_COLOR = (50, 250, 100)


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
    def __init__(self, data_dict: Optional[dict] = None, color=None) -> None:
        super().__init__()
        self._pts: List[Optional[Point]] = []
        self._id = uuid4()
        self._color = color if color is not None else (0, 255, 0)
        self._closed = False
        self.k_mad = 5
        self.contour: np.ndarray = np.array([])
        self.cilium: Optional[dict] = None
        self.ridge: Optional[dict] = None
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

    def draw(self, img, ix_roi: int = 0):
        for ix, pt in enumerate(self._pts[:-1]):
            pt.draw(img, self._color)
            n_pt = self._pts[ix + 1]
            cv2.line(img, (pt.x, pt.y), (n_pt.x, n_pt.y), self._color, 1)
        if len(self._pts) >= 1:
            self._pts[-1].draw(img, self._color)
            cv2.putText(img, str(ix_roi), (self._pts[0].x, self._pts[0].y),
                        cv2.FONT_HERSHEY_PLAIN, 2, self._color)
        if self.closed:
            cv2.line(img, (self._pts[-1].x, self._pts[-1].y), (self._pts[0].x, self._pts[0].y),
                     self._color, 1)
        if len(self.contour) > 0:
            cv2.drawContours(img, [self.contour], 0, CONTOUR_COLOR, 1)
        if self.ridge is not None and 'y' in self.ridge.keys():
            pts = np.vstack((self.ridge['y'], self.ridge['x'])).astype(np.int32).T
            pts_obj = [Point(pt[0], pt[1]) for pt in pts]
            [pt.draw(img, RIDGE_PT_COLOR) for pt in pts_obj]
            cv2.polylines(img, [pts],
                          False, RIDGE_LINE_COLOR, 1)

    def remove_ridge_pt(self, x, y):
        pts_arr = np.vstack((self.ridge['x'], self.ridge['y']))
        ix = self._find_closest(pts_arr.T, x, y)
        self.ridge['x'] = np.delete(self.ridge['x'], ix)
        self.ridge['y'] = np.delete(self.ridge['y'], ix)

    def remove_closest(self, x, y):
        ix = self.closest_point(x, y)
        if ix is not None:
            if ix == 0 or ix == -1:
                self._closed = False
            self.pop(ix)

    def is_point_inside(self, x, y):
        inside = cv2.pointPolygonTest(self._pts_array(), (x, y), False)
        return inside > 0

    @staticmethod
    def _find_closest(pts_arr, x, y):
        dist = np.squeeze(cdist(np.array([(x, y)]), pts_arr))
        return np.argmin(dist)

    def closest_point(self, x, y):
        pts_arr = self._pts_array()
        if len(pts_arr.shape) != 2:
            return None
        return self._find_closest(pts_arr, x, y)

    def _pts_array(self):
        a = np.array([(pt.x, pt.y) for pt in self._pts], dtype=np.int32)
        return a

    @staticmethod
    def _to_dumpable_dict(d: dict):
        if d is None:
            return {}
        new_d = {}
        for k, v in d.items():
            try:
                new_d[k] = v.tolist()
            except AttributeError:
                new_d[k] = v
        return new_d

    @staticmethod
    def _array_from_json(d: dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                v = np.array(v)
            new_d[k] = v
        return new_d

    def to_dict(self):
        cilium = self._to_dumpable_dict(self.cilium)
        ridge = self._to_dumpable_dict(self.ridge)

        d = {'id': str(self.id), 'points': self._pts_array().tolist(),
             'closed': self.closed,
             'contour': self.contour.tolist(),
             'k_mad': self.k_mad,
             'cilium': cilium,
             'ridge': ridge}
        return d

    def from_dict(self, data):
        self._id = data['id']
        self._pts = [Point(pt[0], pt[1]) for pt in data['points']]
        self._closed = data['closed']
        self.k_mad = data['k_mad']
        try:
            self.contour = np.array(data['contour'])
            self.ridge = self._array_from_json(data['ridge'])
            self.cilium = self._array_from_json(data['cilium'])
        except KeyError:
            print('Processing was not done')

    def get_mask(self, img: np.ndarray):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self._pts_array(), 1)
        return mask


class RoiCilium:
    def __init__(self, full_stack, ch_ix: int, msg, img_path):
        self.contour = DrawCiliumContour(full_stack, ch_ix, msg, img_path)


class DrawCiliumContour:
    def __init__(self, full_stack, ch_ix, msg, img_path):
        self.im = full_stack[..., ch_ix].max(0)  # Source image
        self.ch_cil = ch_ix
        self.img_path = Path(img_path)
        self.json_path = self.img_path / (self.img_path.stem + '.json')
        self.full_stack = np.max(full_stack, 0)
        self.full_adj_stack = self.full_stack.copy()
        self.overlays = np.zeros(self.full_stack.shape[-1], dtype=np.bool)
        self.overlays[ch_ix] = True
        self._help = False
        self._help_screen = np.zeros(self.im.shape + (3, ), dtype=np.uint8)
        self._make_help_screen()
        self._display_roi = True
        # self.pts = []  # Coordinates of bounding points
        self._cx = -1
        self._cy = -1
        self._roi_mode = False
        self.k_mad = 5
        self.rois: List[ROI] = []
        self.c_roi: Optional[ROI] = None
        self.all_rois: Optional[List[dict]] = None
        self.handler = msg
        self.im_copy1 = self.im.copy()  # Original copy
        self.im_copy2 = self.im.copy()  # Original copy
        self.immask = np.zeros_like(self.im, dtype=np.uint8)  # Prepare mask
        self.c_mask = np.zeros_like(self.im, dtype=np.uint8)  # Prepare mask
        self.manual_mode = False
        self.th = 110
        self.closest = None
        self.closest_last = None
        with open('colors.json', 'r') as cf:
            self._colors = json.load(cf)
        # Reopen previously saved ROIs
        if self.json_path.exists():
            with open(self.json_path, 'r') as jf:
                json_data = json.load(jf)
                self.rois = [ROI(jd) for jd in json_data]
            if len(self.rois) > 0:
                self.c_roi = self.rois[0]

    def _make_help_screen(self):
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .8
        s, _ = cv2.getTextSize("+TEST", font_face, font_scale, 2)
        print(s)
        with open('roi/help.txt', 'r') as fp:
            for ix, line in enumerate(fp):
                cv2.putText(self._help_screen, line.strip(), (0, (ix+1)*s[1]*2),
                            font_face, font_scale, (235, 235, 235), thickness=2)

    def update_rois(self):
        # Reinitialize image
        self.im_copy1 = np.zeros(self.im_copy1.shape[:2], dtype=np.float32)
        self.full_adj_stack[..., self.ch_cil], saturated = self.apply_th(self.im_copy2.copy(),
                                                                         self.th)
        n_overlays = np.sum(self.overlays)
        for ix, state in enumerate(self.overlays):
            if state:
                self.im_copy1 += self.full_adj_stack[..., ix] / n_overlays
        self.im_copy1 = self.apply_color_map(self.im_copy1)
        self.im_copy1[saturated, ...] = SAT_COLOR
        if self._display_roi:
            for ix, c_roi in enumerate(self.rois):
                c_roi.draw(self.im_copy1, ix)
        if self._help:
            alpha = .7
            self.im_copy1 = cv2.addWeighted(self._help_screen, alpha, self.im_copy1, 1-alpha, 0)
        cv2.imshow(self.handler, self.im_copy1)

    def over_ch(self, state, ch_ix: int):
        self.overlays[ch_ix] = state

    def adjust_th(self, event, ch: int):
        img, _ = self.apply_th(self.full_stack[..., ch].copy(), event)
        self.full_adj_stack[..., ch] = img
        self.update_rois()

    def add_other_channels(self):
        for ch in range(self.full_stack.shape[-1]):
            if ch == self.ch_cil:
                continue
            callback = partial(self.adjust_th, ch=ch)
            cv2.createTrackbar(f'Channel {ch} - Threshold', '', 0, 255, callback)
            cv2.createButton(f'Channel {ch}', self.over_ch, ch, cv2.QT_CHECKBOX, 0)

    def draw_contour(self):
        """
        Main function of class DrawCiliumContour
        """
        cv2.namedWindow(self.handler,
                        flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.handler, 1000, 1000)
        cv2.createTrackbar('Threshold', self.handler, 110, 255, self.callback_trackbar)
        cv2.createTrackbar('k-MAD', self.handler, self.k_mad, 15, self.callback_mad)
        self.add_other_channels()
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
                self._roi_mode = True
                n_color = self._colors[len(self.rois) % len(self._colors)]
                self.c_roi = ROI(color=n_color)
                self.rois.append(self.c_roi)
            elif key == ord('m'):
                self.manual_mode = not self.manual_mode
                self._roi_mode = not self.manual_mode
            elif key == ord('d'):
                ix, r = self._find_roi_under_mouse(self._cx, self._cy)
                if r is not None:
                    self.rois.pop(ix)
                    self.update_rois()
            elif key == ord('e'):
                self._roi_mode = False
                self.c_roi = None
            elif key == ord('c') and self.c_roi is not None and self.c_roi.closed:
                self.c_mask = self.c_roi.get_mask(self.im_copy1)
                self.segment_cilia()
                self.update_rois()
            elif key == ord('x') and self.c_roi is not None:
                # Erase ridge of current ROI
                self.c_roi.ridge = {}
                self.c_roi.cilium = {}
                self.c_roi.contour = np.array([])
            elif key == ord('h'):
                self._help = not self._help
            elif key == ord('r'):
                self._display_roi = not self._display_roi

    def _find_roi_under_mouse(self, x, y):
        """
        Find the roi currently under the mouse cursor

        Parameters
        ----------
        x: float
        y: float

        Return
        ------
        ix: int
            Index in the list of ROIs. Eventually -1 if no roi under mouse
        r: ROI
            ROI object. Eventually None if no ROI under mouse
        """
        for ix, r in enumerate(self.rois):
            if r.is_point_inside(x, y):
                return ix, r
        return -1, None

    def segment_cilia(self):
        g_im = cv2.GaussianBlur(self.im, (5, 5), 5)
        roi_masked = g_im * self.c_mask
        mask_notroi = g_im * (1 - self.c_mask)
        med = np.median(mask_notroi)
        mad = np.median(np.abs(mask_notroi - med))
        # th = np.quantile(roi_masked[roi_masked > 0], .90)
        self.c_roi.k_mad = self.k_mad
        th = med + self.k_mad*mad
        _, th_cil = cv2.threshold(roi_masked, th, 255, cv2.THRESH_BINARY)
        all_cnt, _ = cv2.findContours(th_cil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Keep the biggest object
        cnt = max(all_cnt, key=lambda x: cv2.contourArea(x))
        self.c_roi.contour = np.squeeze(cnt)
        self.c_roi.cilium, self.c_roi.ridge = fit_cilium(self.im, th_cil)
        # cv2.imshow('Thresholded', th_cil)

    def exit(self):
        javabridge.detach()
        self.all_rois = [r.to_dict() for r in self.rois]
        # Useful for testing mostly
        if not self.json_path.parent.is_dir():
            print('WARNING: Data not saved because folder does not exist')
            cv2.destroyWindow(self.handler)
            return
        with open(self.json_path, 'w') as jf:
            json.dump(self.all_rois, jf, indent=2)
        cv2.destroyWindow(self.handler)

    def callback_mad(self, event):
        """
        Callback for the trackbar setting the k for kMAD thresholding
        """
        self.k_mad = cv2.getTrackbarPos('k-MAD', self.handler)

    def callback_trackbar(self, event):
        """
        Callback function responding to trackbar
        Updates working copy of image as a function of threshold
        """
        self.th = cv2.getTrackbarPos('Threshold', self.handler)
        if self.th == 0:
            self.th = 1
        # th_im, _ = self.apply_th(self.im.copy(), self.th)
        # self.im_copy1 = th_im
        # self.im_copy2 = th_im.copy()  # Keep copy without pts and lines
        # cv2.imshow(self.handler, self.im_copy1)
        # self.full_adj_stack[..., self.ch_cil] = th_im
        self.update_rois()

    @staticmethod
    def apply_th(img, th):
        if len(img.shape) == 3:
            bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            bw = img.copy()
        ret, tmp1 = cv2.threshold(bw, th, 255, cv2.THRESH_TRUNC)
        img = (255 * (tmp1.astype('float32') / th)).astype('uint8')
        ret, tmp2 = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
        sat = (tmp2 / 255).astype('uint8')  # Saturated pixels = 1
        # gi = np.where(sat == 1)  # Identify saturated pixels
        gi = sat == 1  # Identify saturated pixels
        # img[gi] = 1
        return img, gi

    @staticmethod
    def apply_color_map(img: np.ndarray):
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        # sat_col = [255, 0, 0]  # Saturated pixels in blue
        # img[gi] = np.tile(sat_col, (np.sum(gi), 1))
        return img
        # for j in range(3):
        #     for i in range(len(gi[0])):
        #         img[gi[0][i], gi[1][i], j] = sat_col[j]
        # self.im_copy2 = self.im_copy1.copy()  # Keep copy without pts and lines
        # cv2.imshow(self.handler, self.im_copy1)
        # return img

    def callback_mouse(self, event, x, y, flags, params):
        self._cx = x
        self._cy = y
        # Editing the current ROI
        if event == cv2.EVENT_LBUTTONDOWN and self.c_roi is not None and self._roi_mode:
            self.c_roi.add_pt(Point(x, y))
            if self.c_roi.closed:
                mask = self.c_roi.get_mask(self.im_copy1)
                self.immask += mask
        elif event == cv2.EVENT_LBUTTONDOWN and self.manual_mode:
            ix, r = self._find_roi_under_mouse(self._cx, self._cy)
            self.c_roi = r
            if self.c_roi.ridge is None:
                self.c_roi.ridge = {'x': [], 'y': [], 'z': []}
            self.c_roi.ridge['x'] = np.hstack((self.c_roi.ridge.get('x', []), y))
            self.c_roi.ridge['y'] = np.hstack((self.c_roi.ridge.get('y', []), x))
            self.c_roi.ridge['z'] = np.hstack((self.c_roi.ridge.get('z', []), self.im[y, x]))
        elif event == cv2.EVENT_LBUTTONDOWN and not self.manual_mode and not self._roi_mode:
            ix, r = self._find_roi_under_mouse(self._cx, self._cy)
            self.c_roi = r
            self._roi_mode = True
        if event == cv2.EVENT_MBUTTONDOWN and self.c_roi is not None:
            if self._roi_mode:
                self.c_roi.remove_closest(x, y)
                # self.im_copy1 = self.im_copy2.copy()  # Reinitialize image
            elif self.manual_mode:
                self.c_roi.remove_ridge_pt(x, y)

        self.update_rois()
        cv2.imshow(self.handler, self.im_copy1)


def fit_cilium(im: np.ndarray, th_cil: np.ndarray):
    """
    Fit a segmented cilium to extract it and its ridge for statistics

    Parameters
    ----------
    im: np.ndarray
        Raw fluorescence z-projection
    th_cil: np.ndarray
        Thresholded image, with the cilium being non-zero elements

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
    # Get the cilium indices
    gi = th_cil > 0
    x, y = np.where(gi)
    # Rectangle around the cilum
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    z = im[xmin:xmax, ymin:ymax]
    # Number of elements in the rectangle
    n = len(z.reshape(-1))
    # Fit a 2D spline on the fluorescence signal with some level of smooth
    bs = RectBivariateSpline(np.arange(xmin, xmax), np.arange(ymin, ymax), z, s=n*5)
    # Get a smoothed version of the cilium
    ze = bs.ev(x, y)
    # Put this in a black image
    fc = np.zeros_like(im)
    fc[x, y] = ze
    # Unique coordinates
    ux = np.unique(x)
    uy = np.unique(y)
    # Maxima for each x and y
    by = np.argmax(fc, 1)[ux]
    bx = np.argmax(fc, 0)[uy]
    # Fit a spline to those maxima, separately depending on whether we use x or y
    sp = UnivariateSpline(ux, by, s=5*len(ux))
    sp2 = UnivariateSpline(uy, bx, s=5*len(ux))
    f_x = sp2(by)
    f_y = sp(bx)
    # Chose the orientation with most points
    if len(ux) > len(uy):
        order = np.argsort(bx)
        xs = bx[order]
        ys = f_y[order]
    else:
        order = np.argsort(by)
        xs = f_x[order]
        ys = by[order]
    # Ridge: Estimated fluorescence value at the coordinates from which the maxima were found
    ridge_z = bs.ev(xs, ys)
    # Length of cilium: lots of tiny triangles
    cil_len = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
    cilium = {'x': x, 'y': y, 'z': ze}
    ridge = {'x': xs, 'y': ys, 'z': ridge_z, 'length': cil_len}

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
    # proj = np.amax(stack[:, :, :, 2], 0)
    my_roi = roi.RoiCilium(stack, 2, 'Set threshold and draw bounding polygon', ROOT_PATH)
    my_roi.contour.draw_contour()
    javabridge.kill_vm()

