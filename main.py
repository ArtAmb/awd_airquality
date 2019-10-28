import statistics as stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import my_stats
import remove as rm
import validator
import view

stats_labels = ["MEAN", "MIN", "MAX", "MEDIAN", "STDEV", "IQR"]

def getValuesFor(summary):
    result = []
    for label in stats_labels:
        result.append(summary[label])

    return result

def showStatsGraph(summaries):
    for summary in summaries:
        values = getValuesFor(summary)
        values.extend(summary["QUANTILES"])
        view_labels = stats_labels.copy()
        view_labels.extend(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
        plot_bar("Stats for " + summary["COLUMN_NAME"], view_labels, values)
        plt.show()


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
    summaries = printColumnStats(columns_names_to_process, data, newData)

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

    showStatsGraph(summaries)

    data.plot.hist()
    plt.show()

    data.boxplot(column=columns_names_to_process)
    plt.show()

    view.showScatterGraph(data, columns_names_to_process[0], columns_names_to_process[1])
    view.showScatterGraph(data, columns_names_to_process[2], columns_names_to_process[3])
    view.showScatterGraph(data, columns_names_to_process[4], columns_names_to_process[0])


def printColumnStats(columns_names_to_process, data, newData):
    summaries = []

    for col in columns_names_to_process:
        summary = {}
        summary["COLUMN_NAME"] = col
        summary["MEAN"] = stats.mean(newData[col])
        summary["MIN"] = min(newData[col])
        summary["MAX"] = max(newData[col])
        summary["MEDIAN"] = stats.median(newData[col])
        summary["STDEV"] = stats.stdev(newData[col])
        summary["IQR"] = my_stats.calculate_iqr(newData[col])
        summary["QUANTILES"] = []

        print("Column: " + col)
        print("Mean with error fields: ", end=" ")
        print(data[col].mean())
        print("Mean without error fields: ", end=" ")
        print(summary["MEAN"])
        print("Min ", end=" ")
        print(summary["MIN"])
        print("Max ", end=" ")
        print(summary["MAX"])
        print("Median ", end=" ")
        print(summary["MEDIAN"])
        print("Standard deviation: ", end=" ")
        print(summary["STDEV"])
        print("IQR: ", end=" ")
        print(summary["IQR"])
        for qua in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print("Quantile[", end="")
            print(qua, end="] ")
            quantilVal = np.quantile(newData[col], qua)
            print(quantilVal)
            summary["QUANTILES"].append(quantilVal)
        print()
        summaries.append(summary)

    return summaries

def plot_bar(title, labels, values):
    index = np.arange(len(labels))
    plt.bar(index, values)
    # plt.xlabel('Genre', fontsize=5)
    # plt.ylabel('No of Movies', fontsize=5)
    plt.xticks(index, labels, fontsize=5, rotation=30)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()
