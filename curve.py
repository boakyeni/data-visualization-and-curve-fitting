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


class Main:
    def __init__(self):
        """
        These functions are only here to create test data
        """

        def exp_decay(x, a, b, c):
            """
            exponential decay
            function: a * exp(-x/b) + c
            a : amplitude
            b : rate
            c : offset
            """
            return a * np.exp(-x / b) + c

        self.xdata = np.linspace(0, 4, 50)
        self.ydata = None
        # y = exp_decay(self.xdata, 2.5, 1.3, 0.5)
        # rng = np.random.default_rng()
        # yerr = 0.2 * np.ones_like(self.xdata)
        # y_noise = yerr * rng.normal(size=self.xdata.size)
        # self.ydata = y + y_noise

        curve_fit_gui(
            None,
            self.xdata,
            self.ydata,
            xlabel="x",
            color="black",
            markerfacecolor="None",
            fitline_color="red",
            title="Complex Graph",
        )


w = Main()
