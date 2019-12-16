import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import second_stage.dataset_devider as dd
from second_stage.dataset_manager import Columns
from second_stage.labels import AirQuality
from second_stage.neural_network import NeutralNetwork


def print_stats(dataset, data_type):
    print(data_type)
    print(dataset[data_type][Columns.LABEL.value].value_counts())


IMPORTANT_COLUMNS = [
    Columns.month,
    Columns.PM10,
    Columns.PM25,
    Columns.O3,
    Columns.NO2,
    Columns.SO2,
    Columns.CO,
    Columns.TEMP,
    Columns.PRES,
    Columns.DEWP,
    Columns.RAIN,
    Columns.WSPM]


def get_expected_output(expected_label):
    result = [0, 0, 0, 0, 0, 0, 0]
    idx = AirQuality[expected_label].value
    result[idx] = 1
    return result


def prepare_inputs(row):
    inputs = []
    for col in IMPORTANT_COLUMNS:
        inputs.append(row[col.value])
    return inputs


PLOT_DATA = [[], [], [], [], [], [], []]


def update_plot_data(errors):
    for x in range(0, AirQuality.__len__()):
        PLOT_DATA[x].append(errors[x])


def main():
    print("Air quality in Changping, one of the Pekin's district, in 2013.03.01 - 2017.02.28")
    data = pd.read_csv("labeled_dataset.csv", delimiter=",")
    dataset = dd.devide_dataset(data)

    print_stats(dataset, "learning")
    print_stats(dataset, "testing")
    print_stats(dataset, "validating")

    inputs_count = IMPORTANT_COLUMNS.__len__()
    outputs_count = AirQuality.__len__()

    neutral_network = NeutralNetwork(inputs_count, [
        [inputs_count, 1],
        [outputs_count, 1]])

    print("START LEARNING")
    for epoch in range(0, 10):
        success_counter = 0
        epoch_result = learning_process(dataset, neutral_network, success_counter)
        print("EPOKA " + str(epoch) + " SUKCESY " + str(success_counter))
        print(epoch_result)
        # visualize_on_plot(epoch_result)

    print("STOP LEARNING")


def visualize_on_plot(epoch_result):
    update_plot_data(epoch_result)
    plt.plot(PLOT_DATA)
    plt.pause(0.05)


def learning_process(dataset, neutral_network, success_counter):
    epoch_result = 0
    COUNTER = {}
    for x in AirQuality:
        COUNTER[x.name] = 0

    for index, row in dataset["learning"].iterrows():
        inputs = prepare_inputs(row)
        outputs = neutral_network.calculate_output(inputs)
        lab = row[Columns.LABEL.value]
        COUNTER[lab] = COUNTER[lab] + 1

        expected_values = get_expected_output(lab)
        errors = np.subtract(expected_values, outputs)
        epoch_result = errors
        # print(errors)
        neutral_network.backpropagation(errors)

        if np.array_equal(expected_values, [round(v) for v in outputs]):
            success_counter = success_counter + 1

    print(COUNTER)
    return epoch_result

        # waga = waga + współczynnik_uczenia * wyjście * błąd

        # print(row[Columns.LABEL.value])
        # print(expected_values)
        # print(outputs)
        # print(np.subtract(expected_values, outputs))
        # print(outputs)


if __name__ == '__main__':
    main()
