import numpy as np
from curvefitgui import curve_fit_gui


class Main:
    def __init__(self):
        """
        This function are only here to create test data
        """

        def cos(x, a, b, c, d):
            return a * np.cos(b * x + c) + d

        self.xdata = np.linspace(0, 4, 50)

        # y data is only set to None because dummy data generated in backend
        # make to return fit result
        # have option to get rid of guess line

        printout = curve_fit_gui(
            None,
            self.xdata,
            None,
            title="Sin Graph",
            xlabel="y axis",
            ylabel="Imaginary",
            fitline_color="red",
            color="black",
            method="leastsq",
            p0=[3, 3, 3, 3, 3],
        )
        # add=[(np.array([1, 2, 3]), np.array([4, 5, 6]), "data2")]


w = Main()

# y = exp_decay(self.xdata, 2.5, 1.3, 0.5)
# rng = np.random.default_rng()
# yerr = 0.2 * np.ones_like(self.xdata)
# y_noise = yerr * rng.normal(size=self.xdata.size)
# self.ydata = y + y_noise
