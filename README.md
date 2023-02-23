source /Users/kojo/opt/anaconda3/bin/activate
conda activate /Users/kojo/opt/anaconda3  

### Getting Started
1. Clone into its own repository (Current working on getting this added to pip, so that it can be installed directly into python environment)
2. Copy and Paste the curvefitgui folder into your project
3. import the gui
    - `from curvefitgui import curve_fit_gui`
4. Call curve_fit_gui:
    - Set first parameter to None to be able to select predefined functions
    - For example: `curve_fit_gui(None,xdata,ydata)`

### curve_fit_gui function:
This starts up the GUI. The call signature is `curve_fit_gui(f,xdata,ydata,xerr=None -> [optional],yerr=None -> [optional],p0=None  -> [optional],xlabel="x-axis" -> [optional],ylabel="y-axis" -> [optional],absolute_sigma=False -> [optional],jac=None -> [optional],showgui=True -> [optional],**kwargs, -> [optional])`
 - f is the fit function which if set to None, gives option of selection predefined fit functions
 - To make a user defined function create a function and pass to curve_fit_gui as f. For example: `def linear(x, a, b,c): return y = a * x + b`

### Customizing Data Plot
The plots are customizable from inside the gui and also from the `curve_fit_gui` function call by using keyword arguments or kwargs. The options available from a matplotlib plot are also available for the main data plot
For example: `curve_fit_gui(None, xdata, ydata, markerfacecolor="None", linestyle="-"`
**For a list of optional keyword arguments visit the matplotlib documentation on this page under the list of available Line2D properties**: [matplotlib.pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
 - Note: Use full names i.e. linewidth not lw
 - In addition title can be set using title keyword argument i.e c(title="More Data")

### customizing Fitline:
Options for customizing fitline inlclude color, linestyle, and thickness. The values are the same from matplotlib, however the argument is preceded with fitline_. For example `curve_fit_gui(None, xdata, ydata, fitline_color="red", fitline_linestle="-")`. The possible customizations for fitline are:
    - fitline_color
    - fitline_linestyle
    - fitline_label
    - fitline_linewidth



To run program
`python3 curve.py"`

## For someone looking to expand this code:
The _tools.py file is the main backend of this program. Here you will find the functionality of the the things seen on the front end.
At current state the program uses custom classes such as FitModel, FitData, FitParameter. These are parts left over from prior to integration with with lmfit model. They were left as they integrate better with the front end. However all the data from those classes can be accessed through the lmfit model. 