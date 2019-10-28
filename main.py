import statistics as stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import my_stats
import remove as rm
import view

columns_names_to_process = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)"]

data = pd.read_csv("AirQualityUCI.csv", delimiter=";")
print("Number of cells")
print(data.size)
print("Removing missing elements....")
data = rm.remove_empty_cells(data)
print("Number of cells")
print(data.size)

print("Removing error elements....")
newData = {}
for col in columns_names_to_process:
    newData[col] = rm.findOkElements(data[col].values)

view.print_title("Stats of columns")

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
    q1 = np.quantile(newData[col], 0.25, interpolation='midpoint')
    q3 = np.quantile(newData[col], 0.75, interpolation='midpoint')
    print("IQR: ", end=" ")
    print(q3 - q1)
    for qua in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("Quantile[", end="")
        print(qua, end="] ")
        print(np.quantile(newData[col], qua, interpolation='midpoint'))
    print()

    col1 = columns_names_to_process[0]
    col2 = columns_names_to_process[1]
    view.print_title("Pearson correlation coefficient")
    my_stats.pearson_correlation(col1, col2, newData)

    view.print_title("Linear regression")
    my_stats.linregress(col1, col2, newData)


plt.show()
#
# view.print_title("Max - min values of columns")
#
# for col in columns_names_to_process:
#     print("Column: " + col)
#     print("Min ", end=" ")
#     print(min(newData[col]))
#     print("Max ", end=" ")
#     print(max(newData[col]))
#     print()
