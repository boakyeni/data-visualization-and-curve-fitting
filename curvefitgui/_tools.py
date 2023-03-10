import inspect
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from lmfit import Model
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit

from ._settings import settings


@dataclass
class FitParameter:
    """
    stores a fitparameter
    """

    name: str
    value: float = 1.0
    sigma: float = 0.0
    fixed: bool = False


@dataclass
class FitModel:
    """
    stores the model
    """

    func: Any
    jac: Any
    weight: str
    # lmfit_model: Model
    # lmfit_parameters: Any
    fitpars: List[FitParameter]
    description: str = ""

    def __post_init__(self) -> None:
        """
        Initializes lmfit model and pararemeters
        """
        self.lmfit_model = Model(self.func)
        """
        parameters and fitpars are the same, 
        however parameters is set up for lmfit, 
        whereas fitpars is connected to the PyQt frontend
        """
        self.parameters = self.lmfit_model.make_params()
        for par in self.fitpars:
            self.parameters[par.name].vary = not par.fixed
            self.parameters[par.name].value = par.value

    def evaluate(self, x):
        return self.func(x, *(par.value for par in self.fitpars))

    def get_numfitpars(self):
        return sum([not par.fixed for par in self.fitpars])

    def update(self):
        for par in self.fitpars:
            self.parameters[par.name].vary = not par.fixed
            self.parameters[par.name].value = par.value


@dataclass
class FitData:
    x: np.array  # x-data
    y: np.array  # y-data
    xe: np.array = None  # error-data on x-values
    ye: np.array = None  # error-data on y-values
    mask: List[bool] = field(init=False)

    def __post_init__(self):
        self.set_mask(-np.inf, np.inf)

    def get(self):
        result = (
            var[self.mask] if var is not None else None
            for var in [self.x, self.y, self.xe, self.ye]
        )
        return result

    def set_mask(self, xmin, xmax):
        self.mask = [xmin <= x <= xmax for x in self.x]

    def get_numfitpoints(self):
        return len(self.x[self.mask])


class Fitter:
    """class to handle the fit"""

    WEIGHTOPTIONS = ("none", "relative", "absolute")

    def __init__(
        self,
        func,
        xdata,
        ydata,
        xerr,
        yerr,
        p0,
        absolute_sigma,
        jac,
        is_complex,
        **kwargs,
    ):

        self.kwargs = kwargs
        self.data = self._init_data(xdata, ydata, xerr, yerr)
        self.model = self._init_model(func, p0, absolute_sigma, jac)
        # Model Result from lmfit
        self.model_result = None
        self.fit_is_valid = False  # becomes True a a valid fit is computed
        self.mean_squared_error = None
        self.fitreport = {}
        self.is_complex = is_complex
        self.method = "leastsq"
        if "method" in kwargs:
            self.method = kwargs["method"]

    def _init_data(self, x, y, xe, ye):

        # validate data
        for var in [x, y]:
            if type(var) is not np.ndarray:
                raise Exception("data should have type numpy array")
        if len(x) != len(y):
            raise Exception("xdata and ydata should be of equal length")

        # get error data if provided
        if ye is not None:
            if type(ye) is not np.ndarray:
                raise Exception("data should have type numpy array")
            if len(ye) != len(y):
                raise Exception("yerr and ydata should be of equal length")

        if xe is not None:
            if type(xe) is not np.ndarray:
                raise Exception("data should have type numpy array")
            if len(xe) != len(x):
                raise Exception("xerr and xdata should be of equal length")

        return FitData(x, y, xe, ye)

    def _init_model(self, func, p0, absolute_sigma, jac):
        # validate function
        if not callable(func):
            raise Exception("Not a valid fit function")

        # determine fitparameters from function
        __args = inspect.signature(func).parameters
        args = [arg.name for arg in __args.values()]

        # set initial values to 1 if not specified
        if p0 is None:
            p0 = [1.0] * len(args[1:])

        # create the fitpars
        fitpars = [
            FitParameter(arg, value) for arg, value in zip(args[1:], p0)
        ]

        # make additional modifications
        if self.data.ye is not None:
            if absolute_sigma:
                weight = self.WEIGHTOPTIONS[2]
            else:
                weight = self.WEIGHTOPTIONS[1]
        else:
            weight = self.WEIGHTOPTIONS[0]

        # create and return the FitModel class
        if func.__doc__ is None:
            description = "no info on model"
        else:
            description = strip_leading_spaces(func.__doc__)

        afitmodel = FitModel(func, jac, weight, fitpars, description)
        return afitmodel

    def fit(self):
        """
        performs the fit
        """
        # prepare model and data p0 are user guesses and pf is whether the guess is fixed or not
        p0 = [fitpar.value for fitpar in self.model.fitpars]
        pF = [fitpar.fixed for fitpar in self.model.fitpars]

        x, y, xe, ye = self.data.get()

        # check number of free fitparameters
        if self.model.get_numfitpars() == 0:
            raise OptimizeWarning(
                "There should be at least one free fitparameter"
            )

        if self._degrees_of_freedom() <= 0:
            raise OptimizeWarning(
                "The number of degrees of freedom (dof) should be at least one."
                + " Try to increase the number of datapoints or to decrease the number of free fitparameters."
            )

        absolute_sigma = self.model.weight == self.WEIGHTOPTIONS[2]
        if self.model.weight == self.WEIGHTOPTIONS[0]:
            ye = np.ones(len(y))  # error of 1 is equal to no weights

        popt, pcov, result = curve_fit_wrapper(
            self.model,
            x,
            y,
            sigma=ye,
            p0=p0,
            pF=pF,
            method=self.method,
            absolute_sigma=absolute_sigma,
            jac=self.model.jac,
        )
        # lmfit model result
        self.model_result = result

        # process results
        self.fit_is_valid = True
        stderrors = np.sqrt(np.diag(pcov))
        for fitpar, value, stderr in zip(self.model.fitpars, popt, stderrors):
            fitpar.value = value
            fitpar.sigma = stderr

        self.mean_squared_error = sum(((y - self.model.evaluate(x)) / ye) ** 2)

        self._create_report()
        return popt, pcov, result

    def get_curve(
        self, xmin=None, xmax=None, numpoints=settings["MODEL_NUMPOINTS"]
    ):
        if xmin is None:
            xmin = self.data.x.min()
        if xmax is None:
            xmax = self.data.x.max()
        xcurve = np.linspace(xmin, xmax, numpoints)
        # check to see if we have fit results yet, so program does not crash
        # in future model should be replaces with lmfit model !!!
        if not self.model_result:
            # ycurve = self.model.evaluate(xcurve)
            ycurve = self.model.lmfit_model.eval(
                self.model.parameters, x=xcurve
            )
        else:
            # lmfit
            ycurve = self.model_result.eval(self.model_result.params, x=xcurve)
        return (xcurve, ycurve)

    def get_model_result(self):
        return self.model_result

    def get_fitcurve(
        self, xmin=None, xmax=None, numpoints=settings["MODEL_NUMPOINTS"]
    ):
        if not self.fit_is_valid:
            return None
        return self.get_curve(xmin, xmax, numpoints)

    def get_residuals(self, check=True):
        """
        return the residuals as y - f(x)
        if check is True(default) only returns values if a valid fit is performed.
        """

        if not self.fit_is_valid and check:
            return None
        if not self.model_result:
            return None
        # return self.data.y - self.model.evaluate(self.data.x)
        # lmfit
        return self.model_result.residual

    def _degrees_of_freedom(self):
        return int(self.data.get_numfitpoints() - self.model.get_numfitpars())

    def _create_report(self):
        def pars_to_dict():
            parsdict = {
                par.name: dict(
                    value=par.value, stderr=par.sigma, fixed=par.fixed
                )
                for par in self.model.fitpars
            }
            return parsdict

        self.fitreport = {
            "FITPARAMETERS": {
                "model": self.model.description,
                "weight": self.model.weight,
                "N": self.data.get_numfitpoints(),
                "dof": self._degrees_of_freedom(),
                "t95-val": stats.t.ppf(0.975, self._degrees_of_freedom()),
            },
            "FITRESULTS": pars_to_dict(),
            "STATISTICS": {
                "Smin, mean squared error": self.mean_squared_error,
                "Chi Squared": self.model_result.chisqr,
                "R^2": self.model_result.rsquared,
            },
        }

    def get_report(self):
        return self.fitreport

    def get_weightoptions(self):
        if self.data.ye is not None:
            return self.WEIGHTOPTIONS
        else:
            # no error data so only one option remains
            return self.WEIGHTOPTIONS[0:1]


def curve_fit_wrapper(
    model, *pargs, p0=None, pF=None, jac=None, method="leastsq", **kwargs
):
    """
    wrapper around the scipy curve_fit() function to allow parameters to be fixed
    same call signature as the curve_fit() function except for:
    pF : 1D numpy array of size n, with n the number of fitparameters of the function
    returns the popt and cov matrices just like the original curve_fit() function
    """

    # extract arguments of the function func
    __args = inspect.signature(model.func).parameters
    args = [arg.name for arg in __args.values()]

    # populate pF (fixed marker) and p0 (param value) to default if not provided in kwargs
    if pF is None:
        pF = np.array([False for _ in args[1:]])  # set all parameters to free
    if p0 is None:
        p0 = np.array([1 for _ in args[1:]])  # set all init values to 1

    # make lists of new function arguments and function arguments to be passed to original function
    newfunc_args = [args[0]] + [
        arg for arg, fix in zip(args[1:], pF) if not fix
    ]
    orifunc_args = [args[0]] + [
        arg if not fix else str(p) for arg, fix, p in zip(args[1:], pF, p0)
    ]

    # make a string defining the new function as a lambda expression and evaluate to function
    fit_func = eval(
        f"lambda {', '.join(newfunc_args)} : func({', '.join(orifunc_args)})",
        locals(),
    )

    # make a string defining the new jacobian function (if specified) as a lambda expression and evaluate to function
    if callable(jac):
        indices = np.array(
            [index for index, value in enumerate(pF) if value == False]
        )
        fit_jac = eval(
            f"lambda {', '.join(newfunc_args)} : jac({', '.join(orifunc_args)})[:, indices]",
            locals(),
        )
    else:
        fit_jac = jac

    # populate a list of initial values for free fit-parameters
    p0_fit = np.array([p for p, fix in zip(p0, pF) if not fix])

    # peform the fit with the reduced function
    # popt, cov = curve_fit(fit_func, *pargs, p0=p0_fit, jac=fit_jac, **kwargs)

    """
    lmfit the code below uses the lmfit model, i keep the old model above as a check
    """

    # gmodel = Model(func)
    # params = gmodel.make_params()
    params = model.parameters
    for arg, fix, p in zip(args[1:], pF, p0):
        params[arg].vary = not fix
        params[arg].value = p
    # From the model wrapper class access lmfit model
    result = model.lmfit_model.fit(
        pargs[1],
        params,
        x=pargs[0],
        calc_covar=True,
        nan_policy="omit",
        method=method,
    )

    popt = np.array(list(result.best_values.values()))
    cov = result.covar

    # rebuild the popt and cov to include fixed parameters
    p0_fix = [p for p, fix in zip(p0, pF) if fix]  # values of fixed parameters
    id_fix = np.where(pF)[0]  # indices of fixed parameters
    """
    No longer needed when using lmfit model
    """
    # for id, p in zip(id_fix, p0_fix):
    #     popt = np.insert(
    #         popt, id, p, axis=0
    #     )  # fill in the popt at the free fit parameters

    # rebuild covariance matrix to include both fixed and optimized pars
    for id in id_fix:
        cov = np.insert(
            cov, id, 0, axis=1
        )  # add zero rows and columns for fixed par
        cov = np.insert(cov, id, 0, axis=0)
    # params here temporarily, in future move lmfit model to fitmodel so that params does not need to be returned here
    return popt, cov, result


def value_to_string(name, value, error, fixed):
    def get_exponent(value):
        """returns the exponent as an int generated by the :.e format specifier"""
        deci = 5
        s = f"{value:.{deci}e}"
        index_sign = s.find("e") + 1
        # Unsure why but sometimes nan prints and program crashes, this prevents it though

        if s[index_sign:] == "nan":
            return int("+00")
        return int(float(s[index_sign:]))

    def float_to_string(value, exponent, sig_digits=2):
        """
        returns a string representation in scientific notation of a number
        specified by value.
        value: the number that is converted
        exponent: the exponent used for the representation
        sig_digits: the number of significant digits that is used
        """
        deci = sig_digits + exponent - get_exponent(value) - 1
        if deci < 0:
            deci = 0
        result1 = f"{value/10**exponent:.{deci}f}"
        result2 = f"{10**exponent:.0e}"[1:]
        return result1, result2

    def to_latex(value_str, exponent, error_str=None):
        """return a latex string
        (value_str +/- error_str) x 10^(exponent)
        checks for nan or None to prevent program crash possibly
        """
        error_str = "0" if error_str is None else error_str

        if error_str and error_str != "nan":
            # != nan added to check for crash possibility
            latex_string = (
                "$= ("
                + value_str
                + "\pm"
                + error_str
                + r")\times$"
                + f"$10^{{{exponent}}}$"
            )
        else:
            latex_string = (
                "$=" + value_str + r"\times$" + f"$10^{{{exponent}}}$"
            )
        return latex_string

    x_e = get_exponent(value)
    if fixed:
        value_str, _ = float_to_string(
            value, x_e, settings["CM_SIG_DIGITS_NO_ERROR"]
        )
        error_str = None
        exponent = x_e
    else:
        dx_e = get_exponent(error)
        if x_e >= dx_e:
            value_str, _ = float_to_string(
                value, x_e, settings["CM_SIG_DIGITS"] + x_e - dx_e
            )
            error_str, _ = float_to_string(
                error, x_e, settings["CM_SIG_DIGITS"]
            )
            exponent = x_e
        else:
            value_str, _ = float_to_string(
                value, dx_e, settings["CM_SIG_DIGITS"] + x_e - dx_e
            )
            error_str, _ = float_to_string(
                error, dx_e, settings["CM_SIG_DIGITS"]
            )
            exponent = dx_e
    combined = name + to_latex(value_str, exponent, error_str)

    return combined


def float_to_str(value, digits):
    """
    return a string reps of a value in scientific notation with the number
    of significant digits specified by digits
    """
    return f"{value:1.{digits}e}"


def strip_leading_spaces(text):
    """removes leading spaces from text"""
    while text.count("\n    ") > 0:
        text = text.replace("\n    ", "\n")
    return text
