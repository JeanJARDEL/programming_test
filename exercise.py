"""
Author : Jean JARDEL
Date : 02/02/2021
"""

import pandas as pd
import numpy as np


# Logic


def smallest_difference(array):
    # Code a function that takes an array and returns the smallest
    # absolute difference between two elements of this array
    # Please note that the array can be large and that the more
    # computationally efficient the better

    # We convert the array into numpy (if it is already numpy it works
    if hasattr(array, "__len__"):
        array = np.array(array)
    else:
        print("ERROR : the argument is not an array")
        return None
    # If the array has only one element, the difference cannot be calculated
    if len(array) <= 1:
        print("WARNING : the array contains less than two values")
        return None
    # We sort the array => complexity in time O(n*log(n))
    array = np.sort(array)
    # We directly return the minimum of all the absolute differences
    return np.abs(np.diff(array)).min()


# Finance and DataFrame manipulation


def macd(prices, window_short=13, window_long=26):
    # Code a function that takes a DataFrame named prices and 
    # returns it's MACD (Moving Average Convergence Difference) as
    # a DataFrame with same shape
    # Assume simple moving average rather than exponential moving average
    # The expected output is in the output.csv file
    res = pd.DataFrame()
    res["date"] = prices.index
    res["MACD"] = prices.rolling(window=window_short).mean() - prices.rolling(window=window_long).mean()
    return res


def sortino_ratio(prices, risk_free_rate=0):
    # Code a function that takes a DataFrame named prices and
    # returns the Sortino ratio for each column
    # Assume risk-free rate = 0
    # On the given test set, it should yield 0.05457

    # We compute all the Sortino ratios of the different columns except for the index column (the date)
    all_ratios = np.zeros(len(prices.columns) - 1)
    i = 0
    for col_name in prices.columns[1:]:
        # Returns of the price DataFrame
        returns = prices[col_name].pct_change().dropna()
        # Adjusted mean of the returns
        mean = np.mean(returns) - risk_free_rate
        # Downside risk (std of negative returns)
        downside_risk = np.std(returns[returns < 0])
        if downside_risk == 0:
            print("ERROR : the downside risk is null")
            return None
        all_ratios[i] = mean / downside_risk
        i += 1
    # FIX ME : it might remain a calculation period to code
    return all_ratios


def expected_shortfall(prices, level=0.95):
    # Code a function that takes a DataFrame named prices and
    # returns the expected shortfall at a given level
    # On the given test set, it should yield -0.03468

    # We first compute the returns of the first column
    first_column = prices.columns[1]
    returns = prices[first_column].pct_change().dropna()
    returns = np.sort(returns)
    # Then we compute the var at the right level
    var = returns[int(len(returns) * (1 - level))]
    # We return the expectation of all the returns below the var
    return np.mean(returns[returns <= var])


# Plot
# We use matplotlib.pyplot but it exists more advanced visualisation tools such as bokeh
# Remark : the import should be done above
import matplotlib.pyplot as plt


def visualize(prices, path):
    # Code a function that takes a DataFrame named prices and
    # saves the plot to the given path
    prices.plot()
    plt.legend()
    plt.savefig(path + 'plot.png')


# Test of the smallest_difference function
print("Smallest difference function : ")
print(smallest_difference([1, 2, 3, 4, 7, 7.5]))
print(smallest_difference(np.array([7.5, 6, 1, 1, 2])))
print(smallest_difference([7, 8, 6, 5.5, 2]))
print(smallest_difference([-2, -8, -10, -1]))
print(smallest_difference([1, -1, -1]))
print(smallest_difference([1]))
print(smallest_difference(12))
print("  ")

# We first define the DataFrame that we will be using
# This path needs to be changed to correspond to project architecture
my_path = "/Users/jean/Desktop/Test Aequam/programming_test/data/"
eur_prices = pd.read_csv(my_path + "data.csv")
print("DataFrame management :")
eur_prices.set_index('date')
print(eur_prices.head())
print("  ")

# Test of MACD function
print(macd(eur_prices).tail())
# There is some differences with the output maybe because of the rolling window
# (that can be adjusted with rolling parameters)
print("  ")

# Test of the Sortino ratio
print(sortino_ratio(eur_prices))
# The difference certainly comes from an investment period adjustment
print("  ")

# Test of the expected shortfall
print(expected_shortfall(eur_prices))

# Test of the visualisation function
visualize(eur_prices, my_path)
# It is possible to display x axis as dates and enable more advanced features with bokeh (and more time)
