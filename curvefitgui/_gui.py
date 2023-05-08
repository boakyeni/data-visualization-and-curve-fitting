# import the required packages
import random
import sys
import warnings
from inspect import signature


import numpy as np
from PyQt5 import QtCore, QtWidgets
from scipy.optimize import OptimizeWarning

from ._settings import settings
from ._tools import Fitter, value_to_string
from ._version import __version__ as CFGversion
from ._widgets import ModelWidget, PlotWidget, ReportWidget


class MainWindow(QtWidgets.QMainWindow):
    """
    Main curve fitting window with data plot, residual plot, and report
    """

    def __init__(self, afitter, xlabel, ylabel, **kwargs):
        super(MainWindow, self).__init__()

        # perform some initial default settings
        """
        We could check here if fitter.model.func is complex, then immediately call fit after initGUI() so that model.result populates 
        """
        self.fitter = afitter
        self.xlabel, self.ylabel = xlabel, ylabel
        self.output = (None, None)
        self.xerrorwarning = settings["XERRORWARNING"]
        self.kwargs = kwargs
        self.OG_model = self.fitter.model  # original model/ user entered model
        self.initGUI(**kwargs)
        # call fit here if complex
        # if self.fitter.is_complex:
        #     self.fit()
        self.plotwidget.update_plot()

    def closeEvent(self, event):
        """needed to properly quit when running in IPython console / Spyder IDE"""
        QtWidgets.QApplication.quit()

    def initGUI(self, **kwargs):
        # main GUI proprieties
        self.setGeometry(100, 100, 1415, 900)
        self.setWindowTitle("curvefitgui " + CFGversion)
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        """
        Holds the fit result has capability of plotting complex result, just needs to know which axis to plot on
        """
        self.model_result = self.fitter.get_model_result()

        # creating the required widgets
        self.plotwidget = PlotWidget(
            self.fitter,
            self.xlabel,
            self.ylabel,
            **kwargs,
        )  # holds the plot

        self.plotwidget.re_init.connect(
            self.refresh
        )  # signal for from update plot button

        self.modelview = ModelWidget(
            self.fitter.model, self.fitter.get_weightoptions()
        )  # shows the model and allows users to set fitproperties
        self.fitbutton = QtWidgets.QPushButton("FIT", clicked=self.fit)
        self.evalbutton = QtWidgets.QPushButton(
            "INITIAL EVALUATION", clicked=self.evaluate
        )
        self.reportview = ReportWidget()  # shows the fitresults
        self.quitbutton = QtWidgets.QPushButton("QUIT", clicked=self.close)

        # create a layout for the buttons
        self.buttons = QtWidgets.QGroupBox()
        buttonslayout = QtWidgets.QHBoxLayout()
        buttonslayout.addWidget(self.evalbutton)
        buttonslayout.addWidget(self.fitbutton)
        self.buttons.setLayout(buttonslayout)

        # create a frame with a vertical layout to organize the modelview, fitbutton and reportview
        self.fitcontrolframe = QtWidgets.QGroupBox()
        self.fitcontrollayout = QtWidgets.QVBoxLayout()
        for widget in (
            self.modelview,
            self.buttons,
            self.reportview,
            self.quitbutton,
        ):
            self.fitcontrollayout.addWidget(widget)
        self.fitcontrolframe.setLayout(self.fitcontrollayout)

        # putting it all together: Setup the main layout
        mainlayout = QtWidgets.QHBoxLayout(self._main)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.plotwidget)
        splitter.addWidget(self.fitcontrolframe)
        mainlayout.addWidget(splitter)

    def showdialog(self, message, icon, info="", details=""):
        """shows an info dialog"""
        msg = QtWidgets.QMessageBox()
        if icon == "critical":
            msg.setIcon(QtWidgets.QMessageBox.Critical)
        if icon == "warning":
            msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setInformativeText(info)
        msg.setWindowTitle("Message")
        msg.setDetailedText(details)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def set_output(self, output):
        """output should be a tuple with variables that are returned when closing the app"""
        self.output = output

    def get_output(self):
        """allows to return the currently stored output of the app when closed"""
        return self.output

    def evaluate(self):
        """updates the model and computes the model curve with the current parameter values"""
        # update the modelvalues from userinput
        try:
            self.modelview.read_values()
        except ValueError:
            self.showdialog(
                "Not a valid input initial parameter values", "critical"
            )
            return None

        # evaluate
        self.reportview.update_report({})
        self.plotwidget.canvas.set_fitline(self.fitter.get_curve())
        self.plotwidget.canvas.set_residuals(
            self.fitter.get_residuals(check=False)
        )
        self.plotwidget.canvas.disable_results_box()
        self.plotwidget.update_plot()

    def fit(self):
        """updates the model performs the fit and updates the widgets with the results"""
        # update the modelvalues from userinput
        try:
            self.modelview.read_values()
        except ValueError:
            self.showdialog(
                "Not a valid input initial parameter values", "critical"
            )
            return None
        # print("selected", self.plotwidget.selected_curve)
        # print("here", self.plotwidget.canvas.ax1.get_lines())

        try:
            if self.plotwidget.selected_curve is not None:
                selected_line = next(
                    x
                    for x in self.plotwidget.canvas.ax1.get_lines()
                    if x.get_label() == self.plotwidget.selected_curve
                )
                # print("sel", selected_line)

                if not self.plotwidget.canvas.fitter.is_complex:
                    (current_x, current_y) = selected_line.get_data()

                else:
                    (current_x, current_y) = self.plotwidget.data_map[
                        selected_line.get_label()
                    ]
                self.fitter.change_data(current_x, current_y, None, None)
                # print("after change", self.fitter.data)

        except Exception as e:
            # print("exp", e)
            self.showdialog("Can't find plot", "critical")
        # update fitrange
        self.plotwidget.canvas.get_range()

        # show warning on xerror data
        if (self.fitter.data.xe is not None) and self.xerrorwarning:
            self.showdialog("The error in x is ignored in the fit!", "warning")
            self.xerrorwarning = False

        # perform the fit
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", OptimizeWarning
            )  # make sure the OptimizeWarning is raised as an exception

            try:
                fitpars, fitcov, result = self.fitter.fit()
            except (ValueError, RuntimeError, OptimizeWarning):
                self.showdialog(str(sys.exc_info()[1]), "critical")

            else:
                """
                change to return result object
                """
                # update output
                self.set_output((fitpars, fitcov, result))

                # update the widgets
                self.modelview.update_values()
                self.reportview.update_report(self.fitter.get_report())
                self.plotwidget.canvas.set_fitline(self.fitter.get_fitcurve())
                self.plotwidget.canvas.set_residuals(
                    self.fitter.get_residuals()
                )
                self.plotwidget.canvas.set_results_box(
                    self._get_result_box_text(), 2
                )
                self.plotwidget.update_plot()

    def _get_result_box_text(self):
        text = "Fit results:"
        text = text + "\n" + "weight:" + self.fitter.model.weight
        for par in self.fitter.model.fitpars:
            n = par.name
            v = par.value
            e = par.sigma
            f = par.fixed
            text = text + "\n" + value_to_string(n, v, e, f)
        return text

    def refresh(self, model, y):
        """
        retrieves data from update signal, plotwidget.re_init(), and updates gui according to user input
        called around line 64

        """
        if model == self.OG_model.func.__name__ or model is None:
            self.fitter.change_model(self.OG_model.func, None, None)
        else:
            self.fitter.change_model(model, None, None)
        self.modelview = ModelWidget(
            self.fitter.model, self.fitter.get_weightoptions()
        )
        # print("refresh", id(self.fitter))

        # self.fitter.data = self.fitter._init_data(x, y, None, None)
        # self.ylabel = "IM"
        # self.fitcontrollayout = QtWidgets.QVBoxLayout()

        child = self.fitcontrollayout.takeAt(0)
        if child.widget():
            child.widget().setParent(None)

        for widget in (
            self.modelview,
            self.buttons,
            self.reportview,
            self.quitbutton,
        ):
            self.fitcontrollayout.addWidget(widget)

        self.fitcontrolframe


# add is_complex parameter
def execute_gui(
    f, xdata, ydata, xerr, yerr, p0, xlabel, ylabel, absolute_sigma, **kwargs
):
    """
    helper function that executes the GUI with an instance of the fitter class
    """

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    is_complex = False if "complex" not in kwargs else kwargs["complex"]

    showgui = True if "showgui" not in kwargs else kwargs["showgui"]

    if not showgui:
        afitter = Fitter(
            f,
            xdata,
            ydata,
            xerr,
            yerr,
            p0,
            absolute_sigma,
            is_complex,
            **kwargs,
        )
        return afitter.fit()

    class CustomDialog(QtWidgets.QDialog):
        """
        choose fit, starting dialog that allows user to choose fit
        """

        def __init__(self):
            super().__init__()

            self.is_complex = (
                False if "complex" not in kwargs else kwargs["complex"]
            )

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
                return a * (np.exp(-(x**2) / b**2)) * np.cos(c * x + d) + e

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
                "y = a * exp(-x^2 / b^2)* cos(c * x + d) + e": decaying_oscillation2,
                "y = a * exp(i * x)": euler,
                "y = a * exp(i * (b * w + c)) + d": complex_function,
            }
            """
            Drop down box of functions for curve fit
            """

            self.setWindowTitle("Choose Fit")
            self.button = QtWidgets.QPushButton("Start")
            self.dialog_layout = QtWidgets.QVBoxLayout()
            message = QtWidgets.QLabel("Choose Curve")

            self.combobox = QtWidgets.QComboBox()

            for item in self.function_map:
                self.combobox.addItem(item)

            # self.combobox.addItem("y = a * exp(-x / b) + c")
            # self.combobox.addItem("y = ax + b")

            self.dialog_layout.addWidget(message)
            self.dialog_layout.addWidget(self.combobox)
            self.dialog_layout.addWidget(self.button)
            self.button.clicked.connect(self.get_text)
            self.button.clicked.connect(self.close)

            self.setLayout(self.dialog_layout)

        def get_text(self):
            """
            On ok from starting pop up this function sets the fit function
            """
            self.is_complex = (
                True if "i" in self.combobox.currentText() else self.is_complex
            )
            self.func = self.function_map[self.combobox.currentText()]

    if f is None:
        dlg = CustomDialog()
        dlg.exec_()
        f = dlg.func
        # Here so that complex bool passes through to widgets
        if dlg.is_complex:
            kwargs["complex"] = True
            is_complex = True

    """
    !!!
    From this line up to sfitter = Fitter(...) will not be needed in final product this
    is just for testing
    !!!
    """

    """
    Create Test Data when user selects function on first screen
    Start with collecting the number of fit parameters there are
    in user selected function

    """
    if ydata is None:
        sig = signature(f)
        params = sig.parameters
        num_of_params = len(params)
        random_arr = [0] * num_of_params
        for i in range(len(random_arr)):
            random_arr[i] = random.uniform(0.5, 4)

        """
        Get correct number of parameters for test data, store function values in array y
        """
        y = f(xdata, *random_arr[1:])
        print(random_arr[1:])

        """
        Create error for test data
        """
        rng = np.random.default_rng()
        test_yerr = 0.2 * np.ones_like(xdata)
        y_noise = test_yerr * rng.normal(size=xdata.size)
        ydata = y + y_noise
        ydata = np.asarray(ydata)

    """
    gonna have to add sigma to this and other places
    """

    afitter = Fitter(
        f,
        xdata,
        ydata,
        xerr,
        yerr,
        p0,
        absolute_sigma,
        is_complex,
        **kwargs,
    )

    # print("original", afitter.data)
    MyApplication = MainWindow(afitter, xlabel, ylabel, **kwargs)
    MyApplication.show()
    app.exec_()
    return MyApplication.get_output()
