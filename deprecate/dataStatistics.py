# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, train data can be accessed directly in the 'input' folder
dataset = pandas.read_csv("./input/train.csv") 

#Drop the first column 'Id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]

# Size of the dataframe

# print(dataset.shape)

# We can see that there are 15120 instances having 55 attributes

#Learning : Data is loaded successfully as dimensions match the data description


# Datatypes of the attributes

# print(dataset.dtypes)

# Learning : Data types of all attributes has been inferred as int64

# Statistical description

pandas.set_option('display.max_columns', None)
# print(dataset.describe())

# Learning :
# No attribute is missing as count is 15120 for all attributes. Hence, all rows can be used
# Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as chi-sq cant be used.
# Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis
# Attributes Soil_Type7 and Soil_Type15 can be removed as they are constant
# Scales are not the same for all. Hence, rescaling and standardization may be necessary for some algos


# Number of instances belonging to each class

# dataset.groupby('Cover_Type').size()

# We see that all classes have an equal presence. No class re-balancing is necessary

print(dataset.skew())