from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def pearson_correlation(col1, col2, newData):
    print(col1 + " x " + col2)

    len = min(newData[col1].__len__(), newData[col2].__len__()) - 1
    val1 = newData[col1][0:len]
    val2 = newData[col2][0:len]
    print(scipy_stats.pearsonr(val1, val2)[0])

def linregress(col1, col2, newData):
    print(col1 + " x " + col2)

    len = min(newData[col1].__len__(), newData[col2].__len__()) - 1
    val1 = newData[col1][0:len]
    val2 = newData[col2][0:len]
    res = scipy_stats.linregress(val1, val2)
    print(res)

    X = val1
    Y = []
    for real_X in val1:
        Y.append(func(real_X, res.slope, res.intercept))

    plt.scatter(val1, val2)
    plt.plot(X, Y, color='red')


def func(x, slope, intercept):
    return slope * x + intercept

def calculate_iqr(values):
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    return calculate_iqr_form_quantiles(q3, q1)

def calculate_iqr_form_quantiles(q3, q1):
    return q3 - q1

def print_outliers(values):
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    iqr = calculate_iqr_form_quantiles(q3, q1)

    bias_low = q1 - 1.5 * iqr
    bias_high = q3 + 1.5 * iqr

    print("Under ", end=" ")
    print(bias_low)
    for v in values:
        if v < bias_low:
            print(v)

    print("Above ", end=" ")
    print(bias_high)
    for v in values:
        if v >= bias_high:
            print(v)



    # X = val1.reshape(-1, 1)
    # Y = val2.reshape(-1, 1)
    # linear_regressor = LinearRegression()
    # linear_regressor.fit(X, Y)
    # Y_pred = linear_regressor.predict(X)