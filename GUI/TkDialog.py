'''
Widget layout (using pack and not grid):

--topf (top frame)--------------------------------------
| Select file button       | Label: selected path      |
--------------------------------------------------------

--middlef (middle frame)--------------------------------
| --left_middlef----------- ---right_middlef---------- |
| | Listbox for selecting | | Listbox for selecting  | |
| | a series              | | channel(s)             | |
| |                       | | + OK button            | |
| ------------------------- -------------------------- |
--------------------------------------------------------

--bottomf (bottom frame)--------------------------------
| --chan1f---- --chan2f---- --chan3f----               |
| | Label    | | Label    | | Label    |               |
| | (image)  | | (image)  | | (image)  |     etc.      |
| | + Scale  | | + Scale  | | + Scale  |               |
| ------------ ------------ ------------               |
--------------------------------------------------------

Actions:
1. Select file button opens lif file and updates list of series
2. List of channels is updated upon series selection
3. Projection images are shown upon channel selection

'''

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import os, re, math, lif
import numpy as np
import utilities as utils
from resizeimage import resizeimage
from pathlib import Path
import javabridge
from typing import Optional
import roi


class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.series_indx = ''  # Index of selected series
        self.selected_chans = []  # Indices of selected channels
        self.selected_chans_mem = []  # To keep track of selected channels in previous step
        self.contains_cilia: Optional[int] = None
        self.width = 1200
        self.height = 800
        self.bg_col = '#ddd'  # Default background color for widgets
        self.button_col = '#fff'  # Default color for buttons
        self.lif_file: Optional[lif.LifClass] = None
        self.build_tkwindow()

    def build_tkwindow(self):
        '''
        Initialize tkinter entry widget
        using layout described above
        '''
        self.title("Select file, series and channels")
        self.geometry(str(self.width) + 'x' + str(self.height))
        self.configure(bg='white')

        # Style and themes: https://tkdocs.com/tutorial/styles.html
        style = ttk.Style()
        style.theme_use('alt')
        style.configure('TButton', background='#fff', foreground='black', width=15, height=12,
                        borderwidth=1, focusthickness=3, focuscolor='none')
        style.map('TButton', background=[('active', '#fff')])

        ### Top frame (topf): select file (Button) and name of selected file (Label)
        # Frame
        self.topf = Frame(self, width=self.width, height=self.height // 5, bg=self.bg_col)
        self.topf.pack(side='top', fill='both', padx=10, pady=5, expand=True)
        self.topf.configure(relief=RAISED, bd=2)
        # Button for selecting file
        self.b_select = ttk.Button(self.topf, text="Select file", command=self.fileDialog)
        self.b_select.pack(side='left', padx=20, pady=5)
        # Label showing fullpath
        self.lab_path = ttk.Label(self.topf)
        self.lab_path.configure(text='', background=self.bg_col, font=("Courier", 14))
        self.lab_path.pack(side='left', padx=0, pady=5)

        ### Middle frame (middlef): Listboxes for chosing series and channels
        # Main frame
        self.middlef = Frame(self, width=self.width, height=self.height // 3, bg='white')
        self.middlef.pack(side='top', fill='both', padx=10, pady=5, expand=True)
        self.middlef.configure(relief=FLAT, bd=2)
        # Frame for left listbox (series)
        self.left_middlef = Frame(self.middlef, width=160, height=self.height // 3, bg=self.bg_col)
        self.left_middlef.pack(side='left', fill='both', padx=0, pady=0, expand=True)
        self.left_middlef.configure(relief=RAISED, bd=2)
        # Label
        self.lab_series = ttk.Label(self.left_middlef)
        self.lab_series.configure(text='Select series', background=self.bg_col,
                                  font=('Helvetica', 14))
        self.lab_series.pack(side='top', anchor='w', padx=20, pady=10)
        # Listbox (lb) for selecting series
        self.lb_series = Listbox(self.left_middlef, selectmode='single', relief=SUNKEN,
                                 exportselection=0)
        self.lb_series.configure(background='white', width=60)
        self.lb_series.pack(side='top', anchor='w', padx=20, pady=5)
        self.lb_series.bind('<<ListboxSelect>>', self.on_series_select)  # Bind to event handler
        # Frame for right listbox (channels)
        self.right_middlef = Frame(self.middlef, width=50, height=self.height // 3, bg=self.bg_col)
        self.right_middlef.pack(side='right', fill='both', padx=20, pady=0, expand=True)
        self.right_middlef.configure(relief=RAISED, bd=2)
        # Label
        self.lab_chan = ttk.Label(self.right_middlef)
        self.lab_chan.configure(text='Select channel(s)', background=self.bg_col,
                                font=('Helvetica', 14))
        self.lab_chan.pack(side='top', anchor='w', padx=20, pady=10)
        # Listbox for selecting channel(s)
        # CHANNELS = ["Channel 1","Channel 2","Channel 3"]
        self.lb_chan = Listbox(self.right_middlef, selectmode='multiple', relief=SUNKEN,
                               exportselection=0)
        self.lb_chan.configure(background='white', width=20)
        # self.lb_chan.insert('end', *CHANNELS)
        self.lb_chan.pack(side='top', anchor='w', padx=20, pady=5)
        self.lb_chan.bind('<<ListboxSelect>>', self.on_chan_select)  # Bind to event handler
        # OK Button
        self.b_ok = ttk.Button(self.right_middlef, text="OK", command=self.run_roi)
        self.b_ok.pack(side='bottom', anchor='w', padx=20, pady=5)

        ### Bottom frame for projection images
        self.bottomf = Frame(self, width=self.width, height=self.height // 3, bg=self.bg_col)
        self.bottomf.pack(side='bottom', fill='both', padx=10, pady=5, expand=True)
        self.bottomf.configure(relief=RAISED, bd=2)

    def run_roi(self):
        # TODO: Open the DrawROI window
        if self.lif_file is None:
            return
        cilia_stack = self.lif_file.get_serie_stack(self.series_indx)
        cilia_proj = cilia_stack[..., self.contains_cilia].max(0)
        my_roi = roi.RoiCilium(cilia_proj, 'Set threshold and draw bounding polygon', self.fullpath)
        my_roi.contour.draw_contour()

    def exit(self):
        self.destroy()
        javabridge.kill_vm()

    def fileDialog(self):
        '''
        Callback function for button select file
        actions:
        1. Extracts info from file path
        and if necessary:
        2. Use Lif class to extract metadata
        3. Compute and save projection images
        '''
        default_path = Path('~').expanduser()
        if (default_path / 'DATA_CILIA').exists():
            default_path = default_path / 'DATA_CILIA'
        self.fullpath = filedialog.askopenfilename(initialdir=default_path.as_posix(),
                                                   title="Select .lif file",
                                                   filetypes=(("lif files", "*.lif"),
                                                              ("all files", "*.*")))
        self.lab_path.configure(text=self.fullpath)
        path_comp = utils.splitall(self.fullpath)  # Get all components of path
        p = re.compile('^20[0-9]{6}$')  # regex for date format yyyymmdd
        res = np.where([bool(p.match(x)) for x in path_comp])[0]  # Which path component matches regex
        if (len(res) == 1):
            self.date = path_comp[int(res)]  # Get date from folder name
        else:
            raise ValueError('Several folders in path match date format yyyymmdd.')
        # Below: use class Lif defined in lif/LifClass.py to handle lif file management
        parent_folder = Path(self.fullpath).parent.parent.parent.as_posix()
        self.lif_file = lif.LifFile(parent_folder, self.date, path_comp[-1])
        self.lif_file.get_metadata(save=True)  # Puts metadata in lif.md
        self.lb_series.delete(0, 'end')  # Clear listbox
        # # Adjust width of listbox to max string length in series names
        series_names = self.lif_file.md['Name'].tolist()
        # len_max = 0
        # for m in series_names:
        #     if len(m) > len_max:
        #         len_max = len(m)
        # self.lb_series.configure(width=len_max)
        # Update listbox with series names
        self.lb_series.insert('end', *series_names)
        self.lif_file.get_proj()
        self.md = self.lif_file.md  # Store metadata (for use in other functions)

    def on_series_select(self, evt):
        '''
        Callback function for event in self.lb_series
        i.e when user selects a series in the listbox
        actions:
        1. Find out number of channels for this series
        2. Create as many sub-frames in bottomf as there are channels
        '''
        w = evt.widget
        index = int(w.curselection()[0])
        self.series_indx = index  # Index of series (for later use)
        self.series_name = w.get(index)
        # Folder name corresponding to selected series:
        folder_name = 'S{:0>2d}'.format(index + 1) + '_' + self.series_name
        # Display number of channels in listbox
        self.nchans = self.md['Nchan'][index]  # Number of channels in series
        CHANNELS = ['Channel ' + str(x + 1) for x in range(self.nchans)]
        self.lb_chan.delete(0, 'end')  # Clear listbox
        self.lb_chan.insert('end', *CHANNELS)
        # Clear content of bottomf
        for w in self.bottomf.winfo_children():
            w.destroy()

        # Dictionnary to dynamically hanle sub-frames instances (one sub-frame/channel)
        lst = ['self.chan' + str(x) + 'f' for x in range(self.nchans)]
        self.subframe_dict = {i: lst[i] for i in range(0, len(lst))}
        # Dictionnary to dynamically handle label instances in sub-frames
        lst = ['self.chan' + str(x) + 'f_label' for x in range(self.nchans)]
        self.subframe_lab_dict = {i: lst[i] for i in range(0, len(lst))}
        # Dictionnary to dynamically handle projection images
        lst = ['self.im_' + str(x) for x in range(self.nchans)]
        self.subframe_im_dict = {i: lst[i] for i in range(0, len(lst))}
        # Dictionnary to dynamically handle scale instances
        # Name is made simple (e.g. self.scale_0 for first channel)
        # So that channel index can be extracted from name of scale instance
        lst = ['self.scale_' + str(x) for x in range(self.nchans)]
        self.subframe_scale_dict = {i: lst[i] for i in range(0, len(lst))}
        # Dictionnary to dynamically handle check buttons and associated booleans
        lst = ['self.check_' + str(x) for x in range(self.nchans)]
        self.subframe_check_dict = {i: lst[i] for i in range(0, len(lst))}
        lst = ['self.booleanvar_' + str(x) for x in range(self.nchans)]
        self.subframe_booleanvar_dict = {i: lst[i] for i in range(0, len(lst))}
        for i in range(len(lst)):
            self.subframe_booleanvar_dict[i] = BooleanVar()

        for i in range(len(lst)):
            # Create sub-frames in bottomf
            self.subframe_dict[i] = Frame(self.bottomf, width=self.width // self.nchans,
                                          bg=self.bg_col)
            self.subframe_dict[i].pack(side='left', fill='both', padx=10, pady=0, expand=True)
            # Grab and resize projection images:
            im_file = os.path.join(os.path.split(self.fullpath)[0], folder_name,
                                   'z_proj_chan' + str(i + 1) + '.png')
            if os.path.exists(im_file):
                im = Image.open(im_file)
                self.subframe_im_dict[i] = resizeimage.resize_width(im, math.floor(
                    0.9 * self.width) // self.nchans)
            # Labels, scales and check buttons will be created upon channel selection by function on_chan_select()

    def on_chan_select(self, evt):
        '''
        Callback function for elf.lb_chan (listbox for selecting channel(s))
        actions:
        1. Checks for changes in list of selected channels
        2. Updates sub-frames by deleting or adding widgets (label and ButtonCheck)
        '''
        # Grab indices of selected channels:
        self.selected_chans = list(evt.widget.curselection())
        # Note: curselection() is a tuple and is converted into a list
        # Folder name corresponding to selected series:
        folder_name = 'S{:0>2d}'.format(self.series_indx + 1) + '_' + self.series_name
        # Check which channels have been (un)selected
        newly_selected = list(set(self.selected_chans) - set(self.selected_chans_mem))
        unselected = list(set(self.selected_chans_mem) - set(self.selected_chans))
        for i in range(self.nchans):
            if i in unselected:
                # Clear content of subframe if channel unselected
                for w in self.subframe_dict[i].winfo_children():
                    w.destroy()
                # Redefining dictionnary entries is necessary here:
                self.subframe_lab_dict[i] = 'self.im_' + str(i)
                self.subframe_scale_dict[i] = 'self.scale_' + str(i)
                self.subframe_check_dict[i] = 'self.check_' + str(i)
            if i in newly_selected:  # Create label and scale in sub-frame
                self.subframe_lab_dict[i] = ttk.Label(self.subframe_dict[i])
                im_file = os.path.join(os.path.split(self.fullpath)[0], folder_name,
                                       'z_proj_chan' + str(i + 1) + '.png')
                if os.path.exists(im_file):
                    # Solution to display image in label (in this order):
                    imtk = ImageTk.PhotoImage(self.subframe_im_dict[i], master=self)
                    self.subframe_lab_dict[i].configure(image=imtk)
                    self.subframe_lab_dict[i].image = imtk
                    self.subframe_lab_dict[i].pack(side='top', fill='both', padx=0, pady=0)
                    # Then use scales to change image saturation level
                    # About scales: https://www.tutorialspoint.com/python/tk_scale.htm
                    # Scale callback configured to supply arguments; solution found here:
                    # https://stackoverflow.com/questions/34684933/how-to-pass-command-to-function-from-tkinter-scale
                    self.subframe_scale_dict[i] = Scale(self.subframe_dict[i], from_=1, to=255,
                                                        orient=HORIZONTAL,
                                                        command=lambda i, name=
                                                        self.subframe_scale_dict[
                                                            i]: self.scale_callback(name, i))
                    self.subframe_scale_dict[i].set(255)  # Initialize scale
                    self.subframe_scale_dict[i].pack(side='top', fill='both', padx=0, pady=0)
                    # Check button below scale
                    self.subframe_check_dict[i] = ttk.Checkbutton(self.subframe_dict[i],
                                                                  text='Contains cilia',
                                                                  variable=
                                                                  self.subframe_booleanvar_dict[i],
                                                                  command=lambda
                                                                      name=self.subframe_check_dict[
                                                                          i]: self.check_callback(
                                                                      name))
                    self.subframe_check_dict[i].pack(side='bottom', fill='both', padx=0, pady=0)
                else:
                    self.subframe_lab_dict[i].configure(text='Image missing',
                                                        background=self.bg_col,
                                                        font=('Helvetica', 14))
                    self.subframe_lab_dict[i].pack(side='top', fill='both', padx=0, pady=0)
        self.selected_chans_mem = self.selected_chans

    def scale_callback(self, name, value):
        '''
        Callback function for scale widgets (sliders for adjusting image contrast)
        The index of the corresponding channel is extracted from scale instance name
        e.g. self.scale_0 for channel 0
        (which is passed as argument)
        Actions:
        1. Identifies channel index
        2. Loads appropriate projection image
        3. Change contrast according to silder value
        4. Display new image in appropriate label
        '''
        i = int(name.split('_')[1])  # Index of channel
        im = Image.fromarray(utils.imLevelHigh(np.array(self.subframe_im_dict[i]), int(value)))
        # Display new image in appropriate label
        imtk = ImageTk.PhotoImage(im, master=self)
        self.subframe_lab_dict[i].configure(image=imtk)
        self.subframe_lab_dict[i].image = imtk

    def check_callback(self, name):
        '''
        Callback function for checkbutton widgets (for indicating which channel contains cilia)
        The index of the corresponding channel is extracted from scale instance name
        e.g. self.check_0 for channel 0
        (which is passed as argument)
        Actions:
        1. Identifies channel index
        2. If checkbutton is on, set others to off (one selection only)
        '''
        i = int(name.split('_')[1])  # Index of channel extracted from name of CheckButton instance
        if self.subframe_booleanvar_dict[i].get():  # If checkbutton is on, put others to off
            for j in list(set(range(self.nchans)) - {int(i)}):
                self.subframe_booleanvar_dict[j].set(False)
            self.contains_cilia = int(i)
        else:
            self.contains_cilia = None
        print(self.contains_cilia)
