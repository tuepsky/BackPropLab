import tkinter as tk
import tkinter.scrolledtext as tkscrolled
import numpy as np
import time
import matplotlib.pyplot as plt
import BplGlobal as g
from BplNeuroNet import NeuronNet


class Training(tk.Frame):
    column_1 = 20
    column_2 = column_1 + 150
    column_3 = column_2 + 110
    column_4 = column_3 + 150
    column_5 = column_4 + 110
    column_6 = column_5 + 140
    column_7 = column_6 + 60

    lineSpace = 40
    header_line = 10
    row_1 = 60

    def run_training(self):
        g.gui.update_status('Training in progress, please wait!')
        # Check parameter
        try:
            alpha = float(self.alpha.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Alpha has an invalid value')
            return
        try:
            epochs = int(self.epochs.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Epochs has an invalid value')
            return
        try:
            hidden_layer_size = int(self.hiddenLayerSize.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Hidden layer size has an invalid value')
            return
        try:
            output_layer_size = int(g.outputLayerSize.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Output layer size has an invalid value')
            return
        try:
            random_seed = int(self.randomSeed.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Random seed has an invalid value')
            return

        if not g.neuronNet:
            num_rows = int(g.numRows.get())
            num_cols = int(g.numCols.get())
            g.neuronNet = NeuronNet(
                num_rows * num_cols,  # = input_layer_size \
                hidden_layer_size,
                output_layer_size,
                random_seed)

        all_pattern = g.allTrainingPattern
        self.all_errors = []
        start_time = time.clock()
        for e in range(epochs):
            self.current_epoch.set(str(e+1))
            self.update()
            errors = g.neuronNet.train(all_pattern, alpha)
            self.all_errors.append(errors)
            elapsed = time.clock() - start_time
            self.training_time.set(int(elapsed))
            self.update()
        self.last_error.set("%6.4f" % self.all_errors[-1])
        g.gui.update_status('')

    def run_test(self):
        if len(g.allTestPattern) == 0:
            tk.messagebox.showinfo('Cannot run test:', 'No test data loaded')
            return

        all_pattern = g.allTestPattern
        pattern_index = 1
        failing_pattern_indexes = []
        for p in all_pattern:
            passed = g.neuronNet.test(p)
            if not passed:
                failing_pattern_indexes.append(str(pattern_index))
            pattern_index += 1
        failure_rate = len(failing_pattern_indexes) / len(all_pattern) * 100
        self.performance.set("%4.2f" % (100 - failure_rate) + "%")
        self.failure_rate.set("%4.2f" % failure_rate + "%")
        failing_pattern = ", ".join(failing_pattern_indexes)
        self.text_value_failing_records.config(state=tk.NORMAL)
        self.text_value_failing_records.delete(1.0, tk.END)
        self.text_value_failing_records.insert(tk.END, failing_pattern)
        self.text_value_failing_records.config(state=tk.DISABLED)

    def show_error_curve(self):
        X = np.linspace(1, len(self.all_errors), len(self.all_errors))
        plt.plot(X, self.all_errors)
        plt.show()

    def reset(self):
        g.neuronNet = None
        self.all_errors = None
        self.current_epoch.set("")
        self.last_error.set("")
        self.training_time.set("")

    def __init__(self, notebook):
        g.numRows = tk.StringVar()
        g.numCols = tk.StringVar()
        g.outputLayerSize = tk.StringVar()
        g.numberTestRecords = tk.StringVar()
        g.numberTrainingRecords = tk.StringVar()
        self.alpha = tk.StringVar()
        self.alpha.set("1")
        self.epochs = tk.StringVar()
        self.epochs.set("50")
        self.randomSeed = tk.StringVar()
        self.randomSeed.set("1")
        self.hiddenLayerSize = tk.StringVar()
        self.hiddenLayerSize.set("20")
        self.neuron_net_initialized = False
        self.all_errors = None
        self.current_epoch = tk.StringVar()
        self.last_error = tk.StringVar()
        self.performance = tk.StringVar()
        self.failure_rate = tk.StringVar()
        self.training_time = tk.StringVar()

        super(Training, self).__init__(notebook, background=g.bgDark)
        self.rows = [self.row_1 + n * self.lineSpace for n in range(10)]

        # Left Column
        lbl_header_left = tk.Label(self, text='Setup', font=g.fontTitle, background=g.bgDark)
        lbl_header_left.place(x=self.column_1, y=self.header_line)

        # Input Layer Width
        lbl_input_layer_width = tk.Label(self, text='Input Layer Width', font=g.fontLabel, background=g.bgDark)
        lbl_input_layer_width.place(x=self.column_1, y=self.rows[0])
        lbl_value_input_layer_width = tk.Label(self, textvariable=g.numCols,
                                               font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_input_layer_width.place(x=self.column_2, y=self.rows[0])

        # Input Layer Height
        lbl_input_layer_height = tk.Label(self, text='Input Layer Height', font=g.fontLabel, background=g.bgDark)
        lbl_input_layer_height.place(x=self.column_1, y=self.rows[1])
        lbl_value_input_layer_height = tk.Label(self, textvariable=g.numRows,
                                                font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_input_layer_height.place(x=self.column_2, y=self.rows[1])

        # Output layer size
        lbl_output_layer_size = tk.Label(self, text='Output Layer Size', font=g.fontLabel, background=g.bgDark)
        lbl_output_layer_size.place(x=self.column_1, y=self.rows[2])
        val_output_layer_size = tk.Label(self, textvariable=g.outputLayerSize, justify=tk.CENTER,
                                         font=g.fontLabel, width=5, background=g.bgLight)
        val_output_layer_size.place(x=self.column_2, y=self.rows[2])

        # Hidden layer size
        lbl_hidden_layer_size = tk.Label(self, text='Hidden Layer Size', font=g.fontLabel, background=g.bgDark)
        lbl_hidden_layer_size.place(x=self.column_1, y=self.rows[3])
        e_hidden_layer_size = tk.Entry(self, textvariable=self.hiddenLayerSize, justify=tk.CENTER,
                                       font=g.fontLabel, width=5, background=g.bgBlue)
        e_hidden_layer_size.place(x=self.column_2, y=self.rows[3])

        # Alpha
        lbl_step_width = tk.Label(self, text='Step Width (alpha)', font=g.fontLabel, background=g.bgDark)
        lbl_step_width.place(x=self.column_1, y=self.rows[4])
        e_alpha = tk.Entry(self, textvariable=self.alpha, justify=tk.CENTER,
                           font=g.fontLabel, width=5, background=g.bgBlue)
        e_alpha.place(x=self.column_2, y=self.rows[4])

        # Epochs
        lbl_epochs = tk.Label(self, text='Epochs', font=g.fontLabel, background=g.bgDark)
        lbl_epochs.place(x=self.column_1, y=self.rows[5])
        e_epochs = tk.Entry(self, textvariable=self.epochs, justify=tk.CENTER,
                            font=g.fontLabel, width=5, background=g.bgBlue)
        e_epochs.place(x=self.column_2, y=self.rows[5])

        # Random Seed
        lbl_random = tk.Label(self, text='Random Seed', font=g.fontLabel, background=g.bgDark)
        lbl_random.place(x=self.column_1, y=self.rows[6])
        e_random = tk.Entry(self, textvariable=self.randomSeed, justify=tk.CENTER,
                            font=g.fontLabel, width=5, background=g.bgBlue)
        e_random.place(x=self.column_2, y=self.rows[6])

        # Middle Column ==============================================================
        lbl_header_middle = tk.Label(self, text='Train', font=g.fontTitle, background=g.bgDark)
        lbl_header_middle.place(x=self.column_3, y=self.header_line)

        # Run button
        button_run = tk.Button(self, text="Run Training", width=21,
                               font=g.fontLabel, background=g.bgDark, command=self.run_training)
        button_run.place(x=self.column_3, y=self.rows[0] - 5)

        # Number of training records
        lbl_failure_rate = tk.Label(self, text='Training Records', font=g.fontLabel, background=g.bgDark)
        lbl_failure_rate.place(x=self.column_3, y=self.rows[1])
        lbl_value_failure_rate = tk.Label(self, textvariable=g.numberTrainingRecords,
                                          font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_failure_rate.place(x=self.column_4, y=self.rows[1])

        # Current Epoch
        lbl_current_epoch = tk.Label(self, text='Current Epoch', font=g.fontLabel, background=g.bgDark)
        lbl_current_epoch.place(x=self.column_3, y=self.rows[2])
        lbl_value_current_epoch = tk.Label(self, textvariable=self.current_epoch,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_current_epoch.place(x=self.column_4, y=self.rows[2])

        # Last error
        lbl_last_error = tk.Label(self, text='Last Error', font=g.fontLabel, background=g.bgDark)
        lbl_last_error.place(x=self.column_3, y=self.rows[3])
        lbl_value_last_error = tk.Label(self, textvariable=self.last_error,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_last_error.place(x=self.column_4, y=self.rows[3])

        # Training duration
        lbl_training_duration = tk.Label(self, text='Time Spent [sec]', font=g.fontLabel, background=g.bgDark)
        lbl_training_duration.place(x=self.column_3, y=self.rows[4])
        lbl_value_training_duration = tk.Label(self, textvariable=self.training_time,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_training_duration.place(x=self.column_4, y=self.rows[4])

        # Please stand by
        self.lbl_please_stand_by = tk.Label(self, text='', fg='yellow', font=g.fontLabel, background=g.bgDark)
        self.lbl_please_stand_by.place(x=self.column_3, y=self.rows[5])

        # Show Error Curve
        button_error_curve = tk.Button(self, text="Show Error Curve", width=21, font=g.fontLabel, background=g.bgDark,
                                       command=self.show_error_curve)
        button_error_curve.place(x=self.column_3, y=self.rows[6])

        # Reset button
        button_reset = tk.Button(self, text="Reset Neural Network", width=21,
                                 font=g.fontLabel, background=g.bgDark, command=self.reset)
        button_reset.place(x=self.column_3, y=self.rows[7])

        # Right Column ==============================================================
        lbl_header_right = tk.Label(self, text='Test', font=g.fontTitle, background=g.bgDark)
        lbl_header_right.place(x=self.column_5, y=self.header_line)

        # Run button
        button_run = tk.Button(self, text="Run Test", width=21,
                               font=g.fontLabel, background=g.bgDark, command=self.run_test)
        button_run.place(x=self.column_5, y=self.rows[0] - 5)

        # Number of test records
        lbl_failure_rate = tk.Label(self, text='Test Records', font=g.fontLabel, background=g.bgDark)
        lbl_failure_rate.place(x=self.column_5, y=self.rows[1])
        lbl_value_failure_rate = tk.Label(self, textvariable=g.numberTestRecords,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_failure_rate.place(x=self.column_6, y=self.rows[1])

        # Performance
        lbl_performance = tk.Label(self, text='Performance', font=g.fontLabel, background=g.bgDark)
        lbl_performance.place(x=self.column_5, y=self.rows[2])
        lbl_value_performance = tk.Label(self, textvariable=self.performance,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_performance.place(x=self.column_6, y=self.rows[2])

        # Failure Rate
        lbl_failure_rate = tk.Label(self, text='Failure Rate', font=g.fontLabel, background=g.bgDark)
        lbl_failure_rate.place(x=self.column_5, y=self.rows[3])
        lbl_value_failure_rate = tk.Label(self, textvariable=self.failure_rate,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_failure_rate.place(x=self.column_6, y=self.rows[3])

        # Failing Records
        lbl_failing_records = tk.Label(self, text='Failing Records:', font=g.fontLabel, background=g.bgDark)
        lbl_failing_records.place(x=self.column_5, y=self.rows[4])

        #scrollbar = tk.Scrollbar(self)
        #scrollbar.place(x=self.column_7, y=self.rows[5])

        self.text_value_failing_records = \
            tkscrolled.ScrolledText(self, font=g.fontLabel, height=20, width=22, background=g.bgLight,
                                    relief=tk.FLAT, state=tk.DISABLED,  wrap=tk.WORD)
        self.text_value_failing_records.place(x=self.column_5, y=self.rows[5])
        #scrollbar.config(command=self.text_value_failing_records.yview)
