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


class CurveFitWindow(QtWidgets.QMainWindow):
    """
    Window for curve fitting app
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window22222")


# p = ReadCommandLineArgs()
# print(p.get_input_data())
# make navigationtoolbar a class and change toolitems to add more buttons
class NavigationToolbar(NavigationToolbar2QT):
    """
    Customized NavigationToolbar
    """

    def __init__(self, figure_canvas, parent=None):
        self.toolitems = (
            ("Home", "Reset original view", "home", "home"),
            ("Back", "Back to  previous view", "back", "back"),
            ("Forward", "Forward to next view", "forward", "forward"),
            (None, None, None, None),
            (
                "Pan",
                "Pan axes with left mouse, zoom with right",
                "move",
                "pan",
            ),
            ("Zoom", "Zoom to rectangle", "zoom_to_rect", "zoom"),
            (
                "Customize Graph",
                "Edit axis, curve and image parameters",
                "grid",
                "edit_parameters",
            ),
            (None, None, None, None),
            (
                "Subplots",
                "Configure subplots",
                "subplots",
                "configure_subplots",
            ),
            ("Save", "Save the figure", "filesave", "save_figure"),
            ("Curve Fit", "Curve Fit", "select", "select_tool"),
        )

        NavigationToolbar2QT.__init__(self, figure_canvas, parent=None)

    def select_tool(self):
        # self.w = CurveFitWindow()
        # self.w.show()

        def func(x, a, b, c):
            """
            exponential decay
            function: a * exp(-b * x) + c
            a : amplitude
            b : rate
            c : offset
            """
            return a * np.exp(-b * x) + c

        # create test data
        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3, 0.5)
        rng = np.random.default_rng()
        yerr = 0.2 * np.ones_like(xdata)
        y_noise = yerr * rng.normal(size=xdata.size)
        ydata = y + y_noise

        # execute the gui
        popt, pcov = curve_fit_gui(
            func, xdata, ydata, yerr=yerr, xlabel="x", ylabel="y"
        )


class MplCanvas(FigureCanvasQTAgg):
    """
    Creates a figure and adds a single set of axes to it.
    Figure extends matplotlib FigureCanvasQTAgg
    Can be embedded straight into an application as any other Qt widget
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        """
        Create the matplotlib FigureCanvas object,
        which defines a single set of axes as self.axes
        """
        input_data = ReadCommandLineArgs()
        sc = MplCanvas(self, width=6, height=4, dpi=100)
        sc.axes.plot(
            input_data.get_input_data()[0],
            input_data.get_input_data()[1],
            "bo",
        )
        """
        Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second
        """
        toolbar = NavigationToolbar(sc, self)
        toolbar.update()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)
        """
        Create a placeholder widget to hold our toolbar and canvas
        """
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
