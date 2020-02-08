'''
BackPropLab-1
Main program
Martin Reiche 2020
'''

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import BplGlobal as g
from BplTraining import Training
from BplInspection import Inspection
import BplFileAccess


class Gui(tk.Tk):
    def __init__(self):
        super(Gui, self).__init__()
        self.title("Back Propagation Lab 1.1 -  www.martin-reiche.de 2020")
        self.minsize(800, 800)

        tab_control = ttk.Notebook(self)
        self.tabTraining = Training(tab_control)
        tab_control.add(self.tabTraining, text='Setup, Train & Test')
        self.tabInspection = Inspection(tab_control)
        tab_control.add(self.tabInspection, text='Inspect')
        tab_control.pack(expand=1, fill='both')

        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load training data from file", command=lambda: self.load_data("train"))
        file_menu.add_command(label="Load test data from file", command=lambda: self.load_data("test"))
        file_menu.add_separator()

        self.test_file_name = tk.StringVar()
        status_bar_1 = tk.Label(self, textvariable=self.test_file_name, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                font=("Arial", 10), pady=4, background=g.bgDark)
        status_bar_1.pack(side=tk.BOTTOM, fill=tk.X)

        self.training_file_name = tk.StringVar()
        status_bar_2 = tk.Label(self, textvariable=self.training_file_name, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                font=("Arial", 10), pady=4, background=g.bgDark)
        status_bar_2.pack(side=tk.BOTTOM, fill=tk.X)
        file_menu.add_command(label="Exit", command=self.quit)

        self.currentStatus = tk.StringVar()
        self.status_bar_3 = tk.Label(self, textvariable=self.currentStatus, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                font=("Arial", 12), pady=4, background=g.bgDark, fg='yellow')
        self.status_bar_3.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, text):
        self.currentStatus.set(text)
        self.status_bar_3.update()

    def load_data(self, p_target):
        files = tk.filedialog.askopenfilenames()
        if len(files) == 0:  # User cancelled
            return

        if p_target == "train":
            g.neuronNet = None
            g.allTestPattern = []
            target = g.allTrainingPattern
            self.training_file_name.set("Training data: " + files[0])
            global_number = g.numberTrainingRecords
            self.update_status('Loading training data, please wait!')
        elif p_target == "test":
            if len(g.allTrainingPattern) == 0:
                tk.messagebox.showinfo('Setup error:', 'Please load training pattern first')
                return
            target = g.allTestPattern
            self.test_file_name.set("Test data: " + files[0])
            global_number = g.numberTestRecords
            self.update_status('Loading test data, please wait!')
        else:
            assert False

        success, number_records = BplFileAccess.load_data_file(files[0], target)
        self.update_status('')
        global_number.set(str(number_records))

        if success and p_target == "test":
            self.tabInspection.create_inspection_display()
            self.tabInspection.apply_pattern(0)
        return


g.gui = Gui()
g.gui.mainloop()
