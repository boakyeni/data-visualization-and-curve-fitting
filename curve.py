import numpy as np

from curvefitgui import curve_fit_gui


class Main:
    def __init__(self):
        """
        This function are only here to create test data
        """

        def exp_decay(x, a, b, c):
            return a * np.exp(-x / b) + c

        self.xdata = np.linspace(0, 4, 50)

        # y data is only set to None because dummy data generated in backend
        # make to return fit result
        # have option to get rid of guess line
        curve_fit_gui(
            None,
            self.xdata,
            None,
            title="Complex Graph",
            xlabel="x axis",
            ylabel="Imaginary",
            fitline_color="red",
            color="black",
        )


w = Main()

# y = exp_decay(self.xdata, 2.5, 1.3, 0.5)
# rng = np.random.default_rng()
# yerr = 0.2 * np.ones_like(self.xdata)
# y_noise = yerr * rng.normal(size=self.xdata.size)
# self.ydata = y + y_noise
