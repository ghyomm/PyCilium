'''
Dialog grid layout:
----------------------------------------------------
|  Select file button    |  Label: selected path   |
----------------------------------------------------
|  Listbox for selecting |  Listbox for selecting  |
|  a series              |  channel(s)             |
----------------------------------------------------
'''

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import os, re
import lif
import numpy as np
import utilities as utils

class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.build_tkwindow()

    def build_tkwindow(self):
        '''Initialize tkinter widgets'''
        self.title("Select file, series and channels")
        self.minsize(800, 300)
        # self.wm_iconbitmap('@icon.xbm')
        self.configure(bg='white')

        ### Top frame (topf): select file (Button) and name of selected file (Label)
        # Frame
        self.topf = Frame(self, width=800, height=100, bg='#eee')
        self.topf.pack(side='top', fill='both', padx=10, pady=5, expand=True)
        self.topf.configure(relief=RAISED, bd=2)
        # Button for selecting file
        self.b_select = ttk.Button(self.topf, text = "Select file",command = self.fileDialog)
        self.b_select.pack(side='left', padx=20, pady=5)
        # Label showing fullpath
        self.lab_path = ttk.Label(self.topf)
        self.lab_path.configure(text = '', background='#eee', font=("Courier", 14))
        self.lab_path.pack(side='left', padx=0, pady=5)

        ### Middle frame (middlef): Listboxes for chosing series and channels
        # Main frame
        self.middlef = Frame(self, width=800, height=300, bg='white')
        self.middlef.pack(side='top', fill='both', padx=10, pady=5, expand=True)
        self.middlef.configure(relief=FLAT, bd=2)
        # Frame for left listbox (series)
        self.left_middlef = Frame(self.middlef, width=160, height=300, bg='#eee')
        self.left_middlef.pack(side='left', fill='both', padx=0, pady=0, expand=True)
        self.left_middlef.configure(relief=RAISED, bd=2)
        # Label
        self.lab_series = ttk.Label(self.left_middlef)
        self.lab_series.configure(text = 'Select series', background='#eee', font=('Helvetica', 14))
        self.lab_series.pack(side='top', anchor='w', padx=20, pady=10)
        # Listbox (lb) for selecting series
        self.lb_series = Listbox(self.left_middlef, selectmode='single', relief=FLAT, exportselection = 0)
        self.lb_series.configure(background='white', width=60)
        self.lb_series.pack(side='bottom', anchor='w', padx=20, pady=5)
        self.lb_series.bind('<<ListboxSelect>>', self.onselect)  # Bind to event handler
        # Frame for right listbox (channels)
        self.right_middlef = Frame(self.middlef, width=50, height=300, bg='#eee')
        self.right_middlef.pack(side='right', fill='both', padx=20, pady=0, expand=True)
        self.right_middlef.configure(relief=RAISED, bd=2)
        # Label
        self.lab_chan = ttk.Label(self.right_middlef)
        self.lab_chan.configure(text = 'Select channel(s)', background='#eee', font=('Helvetica', 14))
        self.lab_chan.pack(side='top', anchor='w', padx=20, pady=10)
        # Listbox for selecting channel(s)
        # CHANNELS = ["Channel 1","Channel 2","Channel 3"]
        self.lb_chan = Listbox(self.right_middlef, selectmode='multiple', relief=FLAT, exportselection = 0)
        self.lb_chan.configure(background='white', width=20)
        # self.lb_chan.insert('end', *CHANNELS)
        self.lb_chan.pack(side='bottom', anchor='w', padx=20, pady=5)

        ### Bottom frame for image
        self.bottomf = Frame(self, width=800, height=300, bg='#eee')
        self.bottomf.pack(side='bottom', fill='both', padx=10, pady=5, expand=True)
        self.bottomf.configure(relief=RAISED, bd=2)
        # Image (will be displayed when selecting series)
        # self.im = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(),'z_proj.png')),master=self)
        self.lab = Label(self.bottomf)
        # self.lab.image = self.im
        # self.lab.configure(image=self.im)
        self.lab.pack(fill='both', padx=0, pady=0)

    def fileDialog(self):
        self.fullpath = filedialog.askopenfilename(initialdir = "/home/ghyomm/DATA_CILIA",
                title = "Select .lif file",filetypes = (("lif files","*.lif"),("all files","*.*")))
        self.lab_path.configure(text = self.fullpath)
        path_comp = utils.splitall(self.fullpath)  # Get all components of path
        p = re.compile('^20[0-9]{6}$')  # regex for date format yyyymmdd
        res = np.where([bool(p.match(x)) for x in path_comp])[0]  # Which path component matches regex
        if(len(res)==1):
            self.date = path_comp[int(res)]  # Get date from folder name
        else:
            sys.exit('Several folders in path match date format yyyymmdd.')
        # Below: use class Lif defined in lif/LifClass.py to handle lif file management
        Lif = lif.LifFile('/home/ghyomm/DATA_CILIA',self.date,path_comp[-1])
        Lif.get_metadata(save=True)  # Puts metadata in lif.md
        self.lb_series.delete(0,'end')  # Clear listbox
        # Adjust width of listbox to max string length in series names
        series_names = Lif.md['Name'].tolist()
        len_max = 0
        for m in series_names:
            if len(m) > len_max:
                len_max = len(m)
        self.lb_series.configure(width=len_max)
        # Update listbox with series names
        self.lb_series.insert('end', *series_names)
        Lif.get_proj()
        self.md = Lif.md  # Store metadata (for use in other functions)

    def onselect(self,evt):
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        # Folder name corresponding to selected series:
        folder_name = 'S{:0>2d}'.format(index+1) + '_' + value
        # Grab projection image from folder and update image in bottom frame
        self.lab.configure(image='')  # Clear label
        proj_im = os.path.join(os.path.split(self.fullpath)[0],folder_name,'z_proj.png')
        self.im = ImageTk.PhotoImage(Image.open(proj_im),master=self)
        self.lab.configure(image=self.im)
        # Also display number of channels in listbox
        nchans = self.md['Nchan'][index]
        CHANNELS = ['Channel ' + str(x+1) for x in range(nchans)]
        self.lb_chan.delete(0,'end')  # Clear listbox
        self.lb_chan.insert('end', *CHANNELS)



root = Root()
root.mainloop()
