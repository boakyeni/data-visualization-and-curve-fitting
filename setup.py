import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

exec(open("./curvefitgui/_version.py").read())

# This call to setup() does all the work
setup(
    name="complex-curve-fit-gui",
    version=__version__,
    description="GUI for lmfit",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/boakyeni/data-visualization-and-curve-fitting",
    author="Kojo Nimako",
    author_email="boakyeni@usc.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <4",
    packages=find_packages("curvefitgui"),
    include_package_data=True,
    package_data={
        "curvefitgui": ["config.txt"],
    },
    # conda
    # install_requires=["matplotlib", "numpy", "scipy", "pyqt", "qtpy"], # need to check versions
    # PyPi
    # install_requires=["matplotlib", "numpy", "scipy", "pyqt5"], # need to check versions
)
