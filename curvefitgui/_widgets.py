# import required packges
from typing import Callable
import warnings
from collections import defaultdict
import io
import string
from sympy.solvers import solve
from sympy import symbols
import re

warnings.filterwarnings("ignore")


import matplotlib.patches as patches
import numpy as np
from lmfit import Model
from matplotlib import rcParams

# Matplotlib packages
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.path import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from ._settings import settings
from ._tools import float_to_str


rcParams["mathtext.fontset"] = "cm"


class DraggableVLine:
    """class to create a draggable vertical line in a plot"""

    lock = None  # we need this to be able to dragg only one line at a time

    def __init__(self, ax, x, linewidth=4, linestyle="--", color="gray"):
        self.line = ax.axvline(
            x=x, linewidth=linewidth, linestyle=linestyle, color=color
        )
        self.press = None
        self.connect()

    def get_pos(self):
        return self.line.get_xdata()[0]

    def remove(self):
        self.line.remove()

    def connect(self):
        "connect to all the events we need"
        self.cidpress = self.line.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def on_press(self, event):
        if event.inaxes != self.line.axes:
            return
        if DraggableVLine.lock is not None:
            return
        contains, _ = self.line.contains(event)
        if not contains:
            return
        x, _ = self.line.get_xdata()
        self.press = x, event.xdata
        DraggableVLine.lock = self

    def on_motion(self, event):
        if self.press is None:
            return
        if DraggableVLine.lock is not self:
            return
        if event.inaxes != self.line.axes:
            return
        x, xpress = self.press
        dx = event.xdata - xpress
        x_clip = x + dx
        self.line.set_xdata([x_clip, x_clip])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        if DraggableVLine.lock is not self:
            return
        DraggableVLine.lock = None
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)


class RangeSelector:
    """class that creates a rangeselector in a plot consisting of two draggable vertical lines"""

    def __init__(self, ax, pos1, pos2):
        self.ax = ax  # axes that holds the lines
        self.pos = [pos1, pos2]  # initial positions of the lines
        self.drag_lines = [DraggableVLine(self.ax, pos) for pos in self.pos]

    def get_range(self):
        pos = [dragline.get_pos() for dragline in self.drag_lines]
        pos.sort()
        return pos

    def remove(self):
        for dragline in self.drag_lines:
            dragline.remove()


class PlotWidget(QtWidgets.QWidget):
    """Qt widget to hold the matplotlib canvas and the tools for interacting with the plots"""

    resized = QtCore.pyqtSignal()  # emits when the widget is resized
    re_init = QtCore.pyqtSignal(
        "PyQt_PyObject", "PyQt_PyObject", "PyQt_PyObject"
    )  # emits when plot is updated

    def __init__(self, fitter, xlabel, ylabel, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = PlotCanvas(fitter, xlabel, ylabel, **kwargs)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.selected_curve = None
        self.initial_complex_curve = (
            self.canvas.ax1.get_lines()[0].get_label()
            if not self.canvas.fitter.is_complex
            else self.canvas.ax1.get_lines()[1].get_label()
        )  # This data is an edge case of selected curve that needs to be saved
        # print("init select", self.selected_curve)
        self.selected_model = None
        self.input_func = None
        self.data_map = {}

        self.hideRes = QtWidgets.QAction("TOGGLE RESIDUAL")
        self.hideRes.setIconText("TOGGLE RESIDUAL")
        self.hideRes.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        self.hideRes.triggered.connect(self.toggle_residual_visibility)

        self.switchData = QtWidgets.QAction("Switch Data")
        self.switchData.setIconText("SELECT DATA")
        self.switchData.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        self.switchData.triggered.connect(self.switch_data)

        self.ACshowselector = QtWidgets.QAction("Activate/Clear RangeSelector")
        self.ACshowselector.setIconText("RANGE SELECTOR")
        self.ACshowselector.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        self.ACshowselector.triggered.connect(self._toggle_showselector)
        self.toolbar.addAction(self.hideRes)

        self.toolbar.addSeparator()

        self.toolbar.addAction(self.switchData)

        self.toolbar.addSeparator()

        self.toolbar.addAction(self.ACshowselector)

        self.toolbar.addSeparator()
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

        self.resized.connect(
            self.update_plot
        )  # update plot when window is resized to fit plot in window

        # functions to switch to
        def linear(x, a, b):
            """
            linear
            function: ax + b
            """

            return a * x + b

        def exp_decay(x, a, b, c):
            """
            exponential decay
            function: a * exp(-x / b) + c
            a : amplitude
            b : rate
            c : offset
            """
            return a * np.exp(-x / b) + c

        def cosine_function(x, a, b, c, d):
            """
            y = a * cos(b * x + c) + d
            """
            return a * np.cos(b * x + c) + d

        def decaying_oscillation(x, a, b, c, d, e):
            """
            y = a * exp(-x / b) * cos(c * x + d) + e
            """
            return a * np.exp(-x / b) * np.cos(c * x + d) + e

        def decaying_oscillation2(x, a, b, c, d, e):
            """
            y = a * exp(-x / b)^2 * cos(c * x + d) + e
            """
            return a * (np.exp(-x / b) ** 2) * np.cos(c * x + d) + e

        def euler(x, a):
            """
            Euler's
            y = a * exp(i * x)
            """
            return a * (np.exp(1j * (x)))

        def complex_function(x, a, b, c, d):
            """
            complex exp
            y = a * exp(i * (b * w + c)) + d
            """
            return a * np.exp(1j * (b * x + c)) + d

        self.function_map = {
            "y = ax + b": linear,
            "y = a * exp(-x / b) + c": exp_decay,
            "y = a * cos(b*x + c) + d": cosine_function,
            "y = a * exp(-x / b) * cos(c * x + d) + e": decaying_oscillation,
            "y = a * exp(-x / b)^2 * cos(c * x + d) + e": decaying_oscillation2,
            "y = a * exp(i * x)": euler,
            "y = a * exp(i * (b * w + c)) + d": complex_function,
        }

    def resizeEvent(self, event):
        self.resized.emit()
        return super(PlotWidget, self).resizeEvent(event)

    def update_plot(self):
        self.canvas.update_plot()

    def _toggle_showselector(self):
        self.canvas.toggle_rangeselector()

    def toggle_residual_visibility(self):
        if not self.canvas.ax2.get_visible():
            self.canvas.ax1.set(xlabel=None)
        else:
            self.canvas.ax1.set_xlabel(
                self.canvas.xlabel,
                fontname=settings["TEXT_FONT"],
                fontsize=settings["TEXT_SIZE"],
            )
        self.canvas.ax2.set_visible(not self.canvas.ax2.get_visible())
        self.canvas.redraw()

    def switch_data(self):
        def save_complex_data():
            """
            before switching curve add complex data to a map of label to x data and y data
            Since complex plots y real and imag, xdata is lost in matplotlib
            """

            if self.canvas.fitter.is_complex:

                if self.selected_curve:
                    self.data_map[self.selected_curve] = (
                        self.canvas.fitter.data.x,
                        self.canvas.fitter.data.y,
                    )
                else:
                    self.data_map[self.initial_complex_curve] = (
                        self.canvas.fitter.data.x,
                        self.canvas.fitter.data.y,
                    )

        def get_selected_curve():
            save_complex_data()
            # print(self.data_map[self.selected_curve])
            self.selected_curve = self.combobox.currentText()

        def get_selected_model():
            self.selected_model = self.model_combobox.currentText()

        # print(self.canvas.ax1.get_lines()[1].get_label())
        self.dialog = QtWidgets.QDialog()
        self.dialog.setWindowTitle("Switch Data")
        self.layout = QtWidgets.QVBoxLayout()

        self.combobox = QtWidgets.QComboBox()
        self.model_combobox = QtWidgets.QComboBox()
        self.user_input = QtWidgets.QLineEdit()
        self.button = QtWidgets.QPushButton("Select")
        self.button.clicked.connect(get_selected_curve)
        self.button.clicked.connect(get_selected_model)
        self.button.clicked.connect(self.re_init_model)
        # self.button.clicked.connect(self.create_callable)
        self.button.clicked.connect(self.dialog.close)

        for item in self.canvas.ax1.get_lines():
            if not re.match(r"fitted curve.*?", item.get_label()):
                self.combobox.addItem(item.get_label())
            # else:
            #     # print(item)

        # for item in self.canvas.ax1.get_lines():
        #     if item.get_label() == "fitted_curve":
        #         self.combobox.addItem(item.get_label())

        self.model_combobox.addItem(
            self.canvas.fitter.orig_model.func.__name__
            if self.canvas.fitter.orig_model.func.__name__
            else "custom_user_function"
        )
        for item in self.function_map:
            self.model_combobox.addItem(item)

        self.layout.addWidget(self.combobox)
        self.layout.addWidget(self.model_combobox)
        # self.layout.addWidget(self.user_input)
        self.layout.addWidget(self.button)

        self.dialog.setLayout(self.layout)
        self.dialog.exec_()

    def re_init_model(self, event):
        new_model = self.canvas.fitter.orig_model.func
        self.canvas.fitter.is_complex = self.canvas.fitter.orig_complex
        if self.selected_model != self.canvas.fitter.orig_model.func.__name__:
            new_model = self.function_map[self.selected_model]
            if "i" in self.selected_model:
                self.canvas.fitter.is_complex = True
            else:
                self.canvas.fitter.is_complex = False
        self.re_init.emit(new_model, None, None)

    # Not fully implemented supposed to allow custom user input during execution
    def custom_user_function(self, *args):
        n = len(args) - 1
        symbol_str = "x" + ":" + str(n)
        return solve(self.input_func, symbols(symbol_str))

    def create_callable(self):
        if len(self.user_input.text()) > 2:
            new_args = ["x"]
            self.input_func = self.user_input.text()
            for c in self.input_func:
                if c in string.ascii_uppercase:
                    new_args.append(c)
            self.function_map[self.input_func] = self.custom_user_function(
                *new_args
            )
            self.model_combobox.addItem(self.input_func)


class PlotCanvas(FigureCanvas):
    """class to hold a canvas with a matplotlib figure and two subplots for plotting data and residuals"""

    def __init__(self, fitter, xlabel, ylabel, **kwargs):
        self.fitter = fitter
        self.xlabel = xlabel
        self.data = fitter.data  # contains the x, y and error data
        self.fitline = None  # contains the fitline if available
        self.residuals = None  # contains the residuals if available
        self.complex_residuals = None
        self.initial_guess_line = None
        self.kwargs = kwargs
        self.additional_data = None
        self.fitline_kwargs = defaultdict(str)
        self.complex = (
            False if "complex" not in self.kwargs else self.kwargs["complex"]
        )
        # print(id(self.fitter))

        # Get Axis titles
        self.ax1_title = (
            self.kwargs["title"] if "title" in self.kwargs else "Data"
        )
        # delete from kwargs to remove matplotlib error
        if "title" in self.kwargs:
            del self.kwargs["title"]
        if "method" in self.kwargs:
            del self.kwargs["method"]
        if "complex" in self.kwargs:
            del self.kwargs["complex"]
        if "add" in self.kwargs:
            self.additional_data = self.kwargs["add"]
            del self.kwargs["add"]
        # Separate Fitline kwargs from data kwargs, at this point kwargs and self.kwargs are same thing
        if "fitline_color" in self.kwargs:
            self.fitline_kwargs["color"] = kwargs["fitline_color"]
            del kwargs["fitline_color"]
        if "fitline_linestyle" in self.kwargs:
            self.fitline_kwargs["linestyle"] = kwargs["fitline_linestyle"]
            del kwargs["fitline_linestyle"]
        if "fitline_linewidth" in self.kwargs:
            self.fitline_kwargs["linewidth"] = kwargs["fitline_linewidth"]
            del kwargs["fitline_linewidth"]
        if "fitline_label" in self.kwargs:
            self.fitline_kwargs["label"] = kwargs["fitline_label"]
            del kwargs["fitline_label"]

        # Sets plots defaults
        self.kwargs = self.set_data_plot_default(**self.kwargs)
        self.fitline_kwargs = self.set_fit_plot_default(**self.fitline_kwargs)

        # setup the FigureCanvas
        self.fig = Figure(dpi=settings["FIG_DPI"], tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(
            self,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        FigureCanvas.updateGeometry(self)

        # init some statevars
        self.range_selector = None

        # create the figure and axes
        gs = self.fig.add_gridspec(3, 1)  # define three rows and one column

        # need to create ax2 first to prevent textbox related to ax1 appear behind residual plot
        self.ax2 = self.fig.add_subplot(
            gs[2, 0]
        )  # ax2 holds the plot of the residuals and spans one row
        self.ax2.title.set_text("Residual Graph")
        self.ax1 = self.fig.add_subplot(
            gs[0:2, 0], sharex=self.ax2
        )  # ax1 holds the plot of the data and spans two rows
        self.ax1.title.set_text(self.ax1_title)

        self.ax1.grid()
        self.ax2.grid()
        self.ax1.set_ylabel(
            ylabel,
            fontname=settings["TEXT_FONT"],
            fontsize=settings["TEXT_SIZE"],
        )
        self.ax2.set_ylabel(
            "residual",
            fontname=settings["TEXT_FONT"],
            fontsize=settings["TEXT_SIZE"],
        )
        self.ax2.set_xlabel(
            xlabel,
            fontname=settings["TEXT_FONT"],
            fontsize=settings["TEXT_SIZE"],
        )

        # create empty lines for the data, fit and residuals
        # (self.data_line,) = self.ax1.plot(
        #     [],
        #     [],
        #     color="black",
        #     marker="o",
        #     fillstyle="none",
        #     lw=0,
        #     label="data",
        # )
        (self.data_line,) = self.ax1.plot(
            [],
            [],
            **self.kwargs,
        )
        (self.fitted_line,) = self.ax1.plot([], [], **self.fitline_kwargs)
        (self.residual_line,) = self.ax2.plot(
            [], [], color="k", marker=".", lw=1
        )
        self.zero_res = None  # holder for a dashed hline to indicate zero in the residual plot
        self.plot_additional_data()
        # create legend
        self.ax1.legend(
            loc="best",
            fancybox=True,
            framealpha=0.5,
            prop={
                "family": settings["TEXT_FONT"],
                "size": settings["TEXT_SIZE"],
            },
        )

        # create an annotate box to hold the fitresults
        bbox_args = dict(
            boxstyle=patches.BoxStyle("round", pad=0.5), fc="0.9", alpha=0.5
        )
        self.result_box = self.ax1.annotate(
            "",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            fontname=settings["TEXT_FONT"],
            size=settings["TEXT_SIZE"],
            bbox=bbox_args,
        )
        self.result_box.draggable()

        """
        self.data.y should be complex still at this point put a print line to check,
        but set_data most likely casts to real so at this point,
        
        """
        # populate plotlines if not complex data since different plot interpretation needed for complex values (the clear circles)
        if not self.fitter.is_complex:
            self.data_line.set_data(self.fitter.data.x, self.fitter.data.y)
        else:
            self.data_line.remove()
            (self.data_line,) = self.plot_complex(
                self.fitter.data.y, "o", **self.kwargs
            )
            self.ax1.legend()

        # create errorbars if required
        if self.data.ye is not None:
            self.yerrobar = self.ax1.errorbar(
                self.data.x,
                self.data.y,
                yerr=self.data.ye,
                fmt="none",
                color=settings["BAR_Y_COLOR"],
                elinewidth=settings["BAR_Y_THICKNESS"],
                capsize=2,
            )
        if self.data.xe is not None:
            self.xerrobar = self.ax1.errorbar(
                self.data.x,
                self.data.y,
                xerr=self.data.xe,
                fmt="none",
                color=settings["BAR_X_COLOR"],
                elinewidth=settings["BAR_X_THICKNESS"],
                capsize=2,
            )

        # set the ticklabel properties
        for labels in [
            self.ax1.get_xticklabels(),
            self.ax1.get_yticklabels(),
            self.ax2.get_xticklabels(),
            self.ax2.get_yticklabels(),
        ]:
            for tick in labels:
                tick.set_color(settings["TICK_COLOR"])
                tick.set_fontproperties(settings["TICK_FONT"])
                tick.set_fontsize(settings["TICK_SIZE"])

        def add_figure_to_clipboard(event):
            if event.key == "ctrl+c":
                with io.BytesIO() as buffer:
                    self.fig.savefig(buffer)
                    QApplication.clipboard().setImage(
                        QImage.fromData(buffer.getvalue())
                    )

        self.fig.canvas.mpl_connect("key_press_event", add_figure_to_clipboard)

    def set_data_plot_default(self, **kwargs):
        kwargs["marker"] = "o" if "marker" not in kwargs else kwargs["marker"]
        kwargs["linewidth"] = (
            0 if "linewidth" not in kwargs else kwargs["linewidth"]
        )
        kwargs["label"] = "data" if "label" not in kwargs else kwargs["label"]
        return kwargs

    def set_fit_plot_default(self, **kwargs):
        kwargs["color"] = "black" if "color" not in kwargs else kwargs["color"]
        kwargs["linestyle"] = (
            "--" if "linestyle" not in kwargs else kwargs["linestyle"]
        )
        kwargs["label"] = (
            "fitted curve" if "label" not in kwargs else kwargs["label"]
        )
        return kwargs

    def set_results_box(self, text, loc):
        self.result_box.set_text(text)

        self.result_box.set_visible(True)

    def disable_results_box(self):
        self.result_box.set_visible(False)

    def toggle_rangeselector(self):
        if self.range_selector is None:
            if self.fitter.is_complex:
                self.range_selector = RangeSelector(
                    self.ax1,
                    np.min(self.fitter.data.y.real),
                    np.max(self.fitter.data.y.real),
                )
            else:
                self.range_selector = RangeSelector(
                    self.ax1,
                    np.min(self.fitter.data.x),
                    np.max(self.fitter.data.x),
                )
            self.redraw()
        else:
            self.range_selector.remove()
            self.range_selector = None
            self.redraw()

    def set_residuals(self, residuals):
        self.residuals = residuals

    def set_fitline(self, fitline):
        """
        Sets both the initial guess fitline and best fit line
        """
        if self.fitter.is_complex and not self.fitter.model_result:
            if not self.initial_guess_line:
                """
                Here is where changes would be made for the initial guess for complex values
                """
                (self.initial_guess_line,) = self.plot_complex(
                    fitline[1], "--g", label="initial guess"
                )
                self.ax1.legend()
        else:
            self.fitline = fitline

    def get_range(self):
        if self.range_selector is None:
            self.fitter.data.set_mask(-np.inf, np.inf)
        else:
            self.fitter.data.set_mask(*self.range_selector.get_range())

    def plot_complex(self, data, *args, **kwargs):
        """
        convenience function for plotting complex quantities
        """
        return self.ax1.plot(data.real, data.imag, *args, **kwargs)

    def create_fitline_kwargs(self, **kwargs):
        fitline_kwargs = defaultdict("str")
        if "fitline_color" in kwargs:
            fitline_kwargs["color"] = kwargs["fitline_color"]
        return fitline_kwargs

    def update_plot(self):
        # update the residuals and/or fitline if present

        if self.fitter.model_result and self.fitter.is_complex:
            # print(id(self.fitter))
            fit_s21 = self.fitter.model_result.eval(
                params=self.fitter.model_result.params, x=self.fitter.data.x
            )
            # clear previous fit line
            # if self.data_line:
            #     self.data_line.remove()
            if self.fitted_line:
                self.fitted_line.remove()
            if self.initial_guess_line:
                self.initial_guess_line.remove()
                self.initial_guess_line = None

            # self.ax2.lines.clear()
            # if self.complex_residuals:
            #     self.complex_residuals.cla()

            # (self.data_line,) = self.plot_complex(self.data.y, **self.kwargs)

            (self.fitted_line,) = self.plot_complex(
                fit_s21, "--k", **self.fitline_kwargs
            )
            """
            Lmfit had wierd residuals fro complex values
            """
            # self.complex_residuals = self.fitter.model_result.plot_residuals(
            #     ax=self.ax2,
            #     datafmt=".-k",
            #     parse_complex="abs",
            #     data_kws={"label": "residual"},
            # )
            if self.residuals is not None:
                # if the zero residual line is not yet created, do so
                if self.zero_res is None:
                    self.ax2.axhline(y=0, linestyle="--", color="black")

                # sort data if required
                if settings["SORT_RESIDUALS"]:
                    order = np.argsort(self.fitter.data.y.real)
                else:
                    order = np.arange(0, len(self.fitter.data.y.real))

                data_copy = np.array([])
                for i in range(len(self.fitter.data.y)):
                    if self.fitter.data.mask[i] == True:
                        data_copy = np.append(
                            data_copy, [self.fitter.data.y.real[i]]
                        )
                self.residual_line.set_data(
                    data_copy[order[: len(self.residuals)]],
                    self.residuals.real[order[: len(self.residuals)]],
                )
            self.ax1.legend()
        else:
            if self.residuals is not None:
                # if the zero residual line is not yet created, do so
                if self.zero_res is None:
                    self.ax2.axhline(y=0, linestyle="--", color="black")

                # sort data if required
                if settings["SORT_RESIDUALS"]:
                    order = np.argsort(self.fitter.data.x)
                else:
                    order = np.arange(0, len(self.fitter.data.x))
                # print("self.data.x", self.data.x, type(self.data.x))
                # print("self.residuals", self.residuals, type(self.residuals))
                # print("order slice", order[: len(self.residuals)], type(order))
                # print("len of mask", len(self.data.mask), type(self.data.mask))
                # print("self mask", self.data.mask)

                """
                Make a copy of data.x
                iterate thru mask
                if data.mask[i] is false then delete data.x[i] in the copy
                use that copy in the residual line set data.
                """
                data_copy = np.array([])
                for i in range(len(self.fitter.data.x)):
                    if self.fitter.data.mask[i] == True:
                        data_copy = np.append(
                            data_copy, [self.fitter.data.x[i]]
                        )
                self.residual_line.set_data(
                    data_copy[order[: len(self.residuals)]],
                    self.residuals[order[: len(self.residuals)]],
                )

                # self.residual_line.set_data(
                #     self.data.x[order],
                #     self.residuals[order[: len(self.residuals)]],
                # )

            if self.fitline is not None:
                self.fitted_line.set_data(self.fitline[0], self.fitline[1])

        # rescale the axis
        self.ax1.relim()
        self.ax1.autoscale()
        self.ax2.relim()
        self.ax2.autoscale()

        # make the min and max yscale limits of the residual plot equal
        ymax = max(np.abs(self.ax2.get_ylim()))
        self.ax2.set_ylim(-ymax, ymax)

        # draw the plot
        self.redraw()

    def plot_additional_data(self):
        if self.additional_data:
            for item in self.additional_data:
                self.ax1.plot(
                    item[0], item[1], label=item[2], marker="o", lw=0
                )

    def redraw(self):
        # self.fig.canvas.draw()
        self.draw()


class ParamWidget(QtWidgets.QWidget):
    """Qt widget to show and change a fitparameter"""

    def __init__(self, par):
        QtWidgets.QWidget.__init__(self)
        self.par = par
        self.label = QtWidgets.QLabel(par.name)
        self.edit = QtWidgets.QLineEdit("")
        self.update_value()
        self.check = QtWidgets.QCheckBox("fix")
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.addWidget(self.check)
        self.setLayout(layout)

    def read_value(self):
        """read userinput (value and fixed) in the parameter data"""
        self.par.value = float(self.edit.text())
        self.par.fixed = self.check.isChecked()
        return None

    def update_value(self):
        value = self.par.value
        self.edit.setText(float_to_str(value, settings["SIGNIFICANT_DIGITS"]))
        return None


class ReportWidget(QtWidgets.QTextEdit):
    """prints a fitreport in a non-editable textbox. Report should be a (nested) dictionary"""

    def __init__(self):
        QtWidgets.QTextEdit.__init__(
            self,
            "none",
        )
        self.setFont(
            QtGui.QFont(settings["REPORT_FONT"], settings["REPORT_SIZE"])
        )
        self.setReadOnly(True)

    def update_report(self, fitreport):
        """updates the text of the texteditbox with the content of a (nested) dictionary fitreport"""

        def print_dict(adict, level):
            for key, item in adict.items():
                if type(item) is dict:
                    if level == 1:
                        self.insertPlainText("========== ")
                    self.insertPlainText(str(key))
                    if level == 1:
                        self.insertPlainText(" ========== ")
                    self.insertPlainText("\n\n")
                    print_dict(item, level + 1)
                else:
                    if type(item) == np.float64:
                        item_str = float_to_str(
                            item, settings["SIGNIFICANT_DIGITS"]
                        )
                    else:
                        item_str = str(item)
                    self.insertPlainText(str(key) + "\t\t: " + item_str + "\n")
            self.insertPlainText("\n")

        self.clear()
        print_dict(fitreport, 1)


class ModelWidget(QtWidgets.QGroupBox):
    """Qt widget to show and control the fit model"""

    def __init__(self, model, weightoptions):
        self.model = model
        QtWidgets.QGroupBox.__init__(self, "Model settings")
        self.initGUI(weightoptions)
        self.set_weight()

    def initGUI(self, weightoptions):
        VBox = QtWidgets.QVBoxLayout()
        HBox = QtWidgets.QHBoxLayout()
        self.parviews = [ParamWidget(par) for par in self.model.fitpars]
        self.WeightLabel = QtWidgets.QLabel("Weighted Fit:")
        self.Yweightcombobox = QtWidgets.QComboBox()
        self.Yweightcombobox.addItems(weightoptions)
        HBox.addWidget(self.WeightLabel)
        HBox.addWidget(self.Yweightcombobox)
        HBox.addStretch(1)
        for parview in self.parviews:
            VBox.addWidget(parview)
        VBox.addLayout(HBox)
        self.setLayout(VBox)
        return None

    def disable_weight(self):
        self.Yweightcombobox.setDisabled(True)

    def enable_weight(self):
        self.Yweightcombobox.setEnabled(True)

    def get_weight(self):
        return self.Yweightcombobox.currentText()

    def set_weight(self):
        index = self.Yweightcombobox.findText(
            self.model.weight, QtCore.Qt.MatchFixedString
        )
        if index >= 0:
            self.Yweightcombobox.setCurrentIndex(index)

    def read_values(self):
        """reads values from userinput into the model"""
        for parview in self.parviews:
            parview.read_value()
        self.model.weight = self.get_weight()
        self.model.update()
        return None

    def update_values(self):
        for parview in self.parviews:
            parview.update_value()
        return None
