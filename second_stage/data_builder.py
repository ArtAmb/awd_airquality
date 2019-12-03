import time

import matplotlib.pyplot as plt
import pandas as pd

import second_stage.labels as labels
from second_stage.dataset_manager import Columns


def clean(data):
    data.dropna(how="all", axis='index', inplace=True)
    data.dropna(how="all", axis='columns', inplace=True)
    data.fillna(inplace=True, method="ffill")
    data.fillna(inplace=True, method="bfill")


def main():
    print("Air quality in Changping, one of the Pekin's district, in 2013.03.01 - 2017.02.28")
    print("Data builder")
    data = pd.read_csv("dataset.csv", delimiter=",")

    print(data.mean())
    clean(data)
    print(data.mean())

    print("Assigning lables START...")
    start = time.time()
    LABEL_COLUMN = []
    for index, row in data.iterrows():
        LABEL_COLUMN.append(labels.classify_row(row).name)

    data[Columns.LABEL.value] = LABEL_COLUMN
    # data[Columns.LABEL.value].value_counts().plot('bar')
    plt.show()

    stop = time.time()
    print("Assigning lables STOP...")
    print("Assigning lables TIME == " + str(stop - start) + " sec")
    data.to_csv("labeled_dataset.csv", sep=',')




if __name__ == '__main__':
    main()
