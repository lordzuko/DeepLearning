# TODO: Add import statements
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('./Data/bmi_and_life_expectancy.csv')
print(bmi_life_data)

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()


# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = None
