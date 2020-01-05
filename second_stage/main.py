import time

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
SUCCESS_PLOT_DATA = []

def update_plot_data(errors):
    for x in range(0, AirQuality.__len__()):
        PLOT_DATA[x].append(errors[x])


class Watchstop:

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return (self.stop_time - self.start_time) * 1000

    def print(self):
        print(str(self.duration()) + ' ms')

    def stop_print_start(self, desc):
        self.stop()
        print (desc + " " + str(self.duration()) + ' ms')
        self.start()


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

    print("PREPARING DATA")
    learning_inputs = get_learning_data(dataset)
    print("START LEARNING")
    ws = Watchstop()
    for epoch in range(0, 3):
        ws.start()
        epoch_result, success_counter = learning_process(neutral_network, learning_inputs)
        print("EPOKA " + str(epoch) + " SUKCESY " + str(success_counter) + " / " + str(dataset["learning"].shape[0]))
        ws.stop_print_start("FULL TIME ==")
        # print(epoch_result)
        # visualize_on_plot(epoch_result)

    # show_errors_plot()
    # plt.plot(SUCCESS_PLOT_DATA)
    # plt.show()

    print("STOP LEARNING")


def show_errors_plot():
    for errs in PLOT_DATA:
        plt.plot(errs)
    plt.show()


def visualize_on_plot(epoch_result):
    update_plot_data(epoch_result)
    plt.plot(PLOT_DATA)
    plt.pause(0.05)

def get_learning_data(dataset):
    result = []
    for index, row in dataset["learning"].iterrows():
        result.append( {
            "input": prepare_inputs(row),
            "label": row[Columns.LABEL.value]})

    return result

def learning_process(neutral_network, input_data):
    epoch_result = 0
    success_counter = 0
    COUNTER = {}
    for x in AirQuality:
        COUNTER[x.name] = 0

    index = 0
    for row in input_data:
        inputs = row["input"].copy()

        outputs = neutral_network.calculate_output(inputs)

        lab = row["label"]
        COUNTER[lab] = COUNTER[lab] + 1

        expected_values = get_expected_output(lab)
        errors = np.subtract(expected_values, outputs)
        epoch_result = errors
        # print(errors)
        neutral_network.backpropagation(errors)

        # print_info(expected_values, outputs)

        if np.array_equal(expected_values, [round(v) for v in outputs]):
            success_counter = success_counter + 1

        if index % 10 == 0:
            SUCCESS_PLOT_DATA.append(success_counter)
            update_plot_data(errors)
        index += 1


    # print(COUNTER)

    return epoch_result, success_counter

    # waga = waga + współczynnik_uczenia * wyjście * błąd

    # print(row[Columns.LABEL.value])
    # print(expected_values)
    # print(outputs)
    # print(np.subtract(expected_values, outputs))
    # print(outputs)


def print_info(expected_values, outputs):
    print("PREDICTED == " + str([round(v) for v in outputs]))
    print("EXPECTED == " + str(expected_values))


if __name__ == '__main__':
    main()
