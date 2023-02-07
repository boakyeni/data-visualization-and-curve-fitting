import sys

import matplotlib
import numpy as np

from curvefitgui import curve_fit_gui

matplotlib.use("Qt5Agg")


from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

"""
Set's up simple Matplotlib canvas FigureCanvasQTAgg which creates
the Figure and adds a single set of axes to it
Canvas objext is also a QWidget so can be embedded into an application
as any other Qt Widget
"""


class ReadCommandLineArgs:
    """
    Functions to read and parse data from command line.
    Current set up takes in two arrays one for x data and one for y data.
    EXAMPLE: "[1,2,3]" "[4,5,6]"
    Will change to make more user friendly soon
    """

    def __init__(self):
        self.data_matrix = []

    def get_input_data(self):
        """
        Returns 2D array of input data points.
        Access x values array with get_input_data()[0],
        y values array with get_input_data[1]
        """
        try:
            argument_x_length = len(sys.argv[1])
            argument_y_length = len(sys.argv[2])
            """
            Can be replaced with merge_input_data to get format [[x1,y1]]
            prep_input_data gives format [[x1,x2], [y1,y2]]
            """
            self.prep_input_data(sys.argv, argument_x_length, "x")
            self.prep_input_data(sys.argv, argument_y_length, "y")
        except:
            print(
                "Please provide input data as two separte arrays, one for x coordinates and one for y coordinates"
            )
            sys.exit(1)
        return self.data_matrix

    def merge_input_data(self, argument, argument_length, coordinate):
        """
        Takes as parameters sys.args, length of coordinate array, and either
        a string of x or y to indicate which coordinate array, then merges
        data into 2D array of format [[x1,y1], [x2,y2]]
        """
        if coordinate == "x":
            coordinate = 1
        else:
            coordinate = 2
        parsed_argument_list = argument[coordinate][
            1 : argument_length - 1
        ].split(",")
        """
        initialize data matrix as Nx2 array with all zeros
        """
        if not self.data_matrix:
            self.data_matrix = [
                [0] * 2 for i in range(len(parsed_argument_list))
            ]

        if len(self.data_matrix) != len(parsed_argument_list):
            print(
                "Please make sure there are equal number of x and y data points"
            )
            sys.exit(1)

        """
        Coordinate - 1 equals 0 or 1 for x and y arrays respectively
        """
        for index, el in enumerate(parsed_argument_list):
            self.data_matrix[index][coordinate - 1] = int(el)

    def prep_input_data(self, argument, argument_length, coordinate):
        """
        Takes as parameters sys.args, length of coordinate array, and either
        a string of x or y to indicate which coordinate array, then merges
        data into 2D array of format [[x1,x2], [y1,y2]]
        """
        if coordinate == "x":
            coordinate = 1
        else:
            coordinate = 2
        parsed_argument_list = argument[coordinate][
            1 : argument_length - 1
        ].split(",")
        """
        initialize data matrix as Nx2 array with all zeros
        """
        if not self.data_matrix:
            self.data_matrix = [
                [0] * len(parsed_argument_list) for i in range(2)
            ]
        """
        Coordinate - 1 equals 0 or 1 for x and y arrays respectively
        """
        for index, el in enumerate(parsed_argument_list):
            self.data_matrix[coordinate - 1][index] = int(el)


# p = ReadCommandLineArgs()
# print(p.get_input_data())
# make navigationtoolbar a class and change toolitems to add more buttons


# class MplCanvas(FigureCanvasQTAgg):
#     """
#     Creates a figure and adds a single set of axes to it.
#     Figure extends matplotlib FigureCanvasQTAgg
#     Can be embedded straight into an application as any other Qt widget
#     """

#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super(MplCanvas, self).__init__(fig)


# class Functions:
#     """
#     Dropdown menu of fit functions
#     """

#     def __init__(self):
#         super().__init__()
#         self.function_map = {
#             "y = ax + b": linear,
#             "y = a * exp(-b * x) + c": exp_decay,
#         }

#         self.combobox = QtWidgets.QComboBox()
#         self.combobox.addItem("y = ax + b")
#         self.combobox.addItem("y = a * exp(-b * x) + c")

#         def linear(x, a, b):

#             return a * x + b

#         def exp_decay(x, a, b, c):
#             """
#             exponential decay
#             function: a * exp(-b * x) + c
#             a : amplitude
#             b : rate
#             c : offset
#             """
#             return a * np.exp(-b * x) + c

#         return self.function_map[self.combobox.currentText()]


class Main:
    def __init__(self):
        # super().__init__()
        # self.title = "Select Fit Function"
        # self.top = 100
        # self.left = 100
        # self.width = 680
        # self.height = 500
        """
        These functions are only here to create test data
        """

        # def linear(x, a, b):

        #     return a * x + b

        def exp_decay(x, a, b, c):
            """
            exponential decay
            function: a * exp(-x/b) + c
            a : amplitude
            b : rate
            c : offset
            """
            return a * np.exp(-x / b) + c

        # def cosine_function(x, a, b, c, d):
        #     """
        #     y = a * cos(b * x + c) + d
        #     """
        #     return a * np.cos(b * x + c) + d

        # def decaying_oscillation(x, a, b, c, d, e):
        #     """
        #     y = a * exp(-x / b)
        #     """
        #     return a * np.exp(-x / b) * np.cos(c * x + d) + e

        # def decaying_oscillation2(x, a, b, c, d, e):
        #     """ """
        #     return a * (np.exp(-x / b) ** 2) * np.cos(c * x + d) + e

        # self.function_map = {
        #     "y = ax + b": linear,
        #     "y = a * exp(-x/b) + c": exp_decay,
        #     "y = a * cos(b*x + c) + d": cosine_function,
        #     "y = a * exp(-x / b) * cos(c * x + d) + e": decaying_oscillation,
        #     "y = a * exp(-x / b)^2 * cos(c * x + d) + e": decaying_oscillation2,
        # }

        # self.combobox = QtWidgets.QComboBox()
        # self.combobox.addItem("y = a * exp(-b * x) + c")
        # self.combobox.addItem("y = ax + b")

        # self.func = self.function_map["y = a * exp(-x/b) + c"]

        self.xdata = np.linspace(0, 4, 50)
        y = exp_decay(self.xdata, 2.5, 1.3, 0.5)
        rng = np.random.default_rng()
        yerr = 0.2 * np.ones_like(self.xdata)
        y_noise = yerr * rng.normal(size=self.xdata.size)
        self.ydata = y + y_noise

        curve_fit_gui(None, self.xdata, self.ydata)

        # self.pushButton = QtWidgets.QPushButton("Start", self)
        # self.pushButton.setToolTip("<h3>Start the Session</h3>")

        # self.pushButton.clicked.connect(self.curve_fit_window)

        # self.main_window(self.combobox, self.pushButton)

        # define x and y data as 1 dimensional numpy arrays of equal length

        # execute the function

    # def main_window(self, combo, button):
    #     """
    #     Starting Window to select curve fit function
    #     """
    #     self.label = QtWidgets.QLabel("Select a Fit Function", self)
    #     self.dropdown = combo
    #     self.start = button

    #     layout = QtWidgets.QVBoxLayout()
    #     layout.addWidget(self.dropdown)
    #     layout.addWidget(self.start)
    #     """
    #     Create a placeholder widget to hold function select and start button
    #     """
    #     widget = QtWidgets.QWidget()
    #     widget.setLayout(layout)
    #     self.setCentralWidget(widget)

    #     self.setWindowTitle(self.title)
    #     self.setGeometry(self.top, self.left, self.width, self.height)
    #     self.show()

    # def curve_fit_window(self):

    #     self.hide()


# app = QtWidgets.QApplication(sys.argv)
# w = Main()
# app.exec_()

w = Main()
