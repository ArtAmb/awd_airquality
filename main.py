import statistics as stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import validator

import my_stats
import remove as rm
import view


def main():
    columns_names_to_process = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)"]
    data = pd.read_csv("AirQualityUCI.csv", delimiter=";")
    print("Number of cells")
    print(data.size)
    print("Removing missing elements....")
    data = rm.remove_empty_cells(data)
    print("Number of cells")
    print(data.size)
    print("Removing error elements....")

    dataValidator = validator.RowValidator(columns_names_to_process);

    for col in columns_names_to_process:
        indexes_to_drop = data[data[col] < 0].index
        data.drop(indexes_to_drop, inplace=True)
    print(data.size)

    newData = {}
    for col in columns_names_to_process:
        newData[col] = rm.findOkElements(data[col].values)
    view.print_title("Stats of columns")
    printColumnStats(columns_names_to_process, data, newData)

    view.print_title("Outliners")
    for col in columns_names_to_process:
        print("Column: " + col)
        my_stats.print_outliers(newData[col])

    col1 = columns_names_to_process[0]
    col2 = columns_names_to_process[1]
    view.print_title("Pearson correlation coefficient")
    my_stats.pearson_correlation(col1, col2, newData)

    view.print_title("Linear regression")
    my_stats.linregress(col1, col2, newData)
    plt.show()

    data.plot.hist()
    plt.show()

    data.boxplot(column = columns_names_to_process)
    plt.show()

    view.showScatterGraph(data, columns_names_to_process[0], columns_names_to_process[1])
    view.showScatterGraph(data, columns_names_to_process[2], columns_names_to_process[3])
    view.showScatterGraph(data, columns_names_to_process[4], columns_names_to_process[0])


def printColumnStats(columns_names_to_process, data, newData):
    for col in columns_names_to_process:
        print("Column: " + col)
        print("Mean with error fields: ", end=" ")
        print(data[col].mean())
        print("Mean without error fields: ", end=" ")
        print(stats.mean(newData[col]))
        print("Min ", end=" ")
        print(min(newData[col]))
        print("Max ", end=" ")
        print(max(newData[col]))
        print("Median ", end=" ")
        print(stats.median(newData[col]))
        print("Standard deviation: ", end=" ")
        print(stats.stdev(newData[col]))
        print("IQR: ", end=" ")
        print(my_stats.calculate_iqr(newData[col]))
        for qua in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print("Quantile[", end="")
            print(qua, end="] ")
            print(np.quantile(newData[col], qua))
        print()


if __name__ == '__main__':
    main()
