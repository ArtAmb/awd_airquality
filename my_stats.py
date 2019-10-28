from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
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


    # X = val1.reshape(-1, 1)
    # Y = val2.reshape(-1, 1)
    # linear_regressor = LinearRegression()
    # linear_regressor.fit(X, Y)
    # Y_pred = linear_regressor.predict(X)