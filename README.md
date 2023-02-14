source /Users/kojo/opt/anaconda3/bin/activate
conda activate /Users/kojo/opt/anaconda3  

To run program
`python3 curve.py"`

## For someone looking to expand this code:
The _tools.py file is the main backend of this program. Here you will find the functionality of the the things seen on the front end.
At current state the program uses custom classes such as FitModel, FitData, FitParameter. These are parts left over from prior to integration with with lmfit model. They were left as they integrate better with the front end. However all the data from those classes can be accessed through the lmfit model. 