import matplotlib.pyplot as plt
import numpy as np

from second_stage.dataset_manager import Columns
from second_stage.labels import AirQuality

PLOT_DATA = [[], [], [], [], [], [], []]
SUCCESS_PLOT_DATA = []

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


def show_success_plot():
    plt.plot(SUCCESS_PLOT_DATA)
    plt.show()


def show_some_errors(qualities):
    for air_quality in qualities:
        plt.plot(PLOT_DATA[air_quality.value])
    plt.show()


def show_important_errors():
    show_some_errors([
        AirQuality.EXCELLENT,
        AirQuality.GOOD,
        AirQuality.MODERATELY_POLLUTED,
        AirQuality.LIGHTLY_POLLUTED])


def show_not_important_errors():
    show_some_errors([
        AirQuality.HEAVILY_POLLUTED,
        AirQuality.UNDEFINED,
        AirQuality.SEVERELY_POLLUTED])


def show_errors_plot():
    for errs in PLOT_DATA:
        plt.plot(errs)
    plt.show()


def visualize_on_plot(epoch_result):
    update_plot_data(epoch_result)
    plt.plot(PLOT_DATA)
    plt.pause(0.05)


def update_plot_data(errors):
    for x in range(0, AirQuality.__len__()):
        PLOT_DATA[x].append(errors[x])


def get_expected_output(expected_label):
    result = [0, 0, 0, 0, 0, 0, 0]
    idx = AirQuality[expected_label].value
    result[idx] = 1
    return result

def change_value(val, max):
    if val == max:
        return 1
    else:
        return 0

def start_process(neutral_network, input_data, learning):
    epoch_result = 0
    success_counter = 0
    COUNTER = {}
    for x in AirQuality:
        COUNTER[x.name] = 0

    index = 0
    for row in input_data:
        inputs = row["input"].copy()
        lab = row["label"]
        COUNTER[lab] = COUNTER[lab] + 1

        outputs = neutral_network.calculate_output(inputs)
        expected_values = get_expected_output(lab)
        # print_info(expected_values, outputs)

        max_v = max(outputs)
        new_outs = [change_value(o, max_v) for o in outputs]
        if np.array_equal(expected_values, new_outs):
            success_counter += 1
        # else:
        #     print("REAL == " + str(outputs))
        #     print("ROUNDED == " + str(new_outs))
        #     print_info(expected_values, outputs)

        errors = np.subtract(expected_values, outputs)
        epoch_result = errors

        if learning:
            neutral_network.backpropagation(errors)

        # if index % 10 == 0:
        #     SUCCESS_PLOT_DATA.append(success_counter)

        index += 1
        if learning:
            update_plot_data(epoch_result)

    # print(COUNTER)

    return epoch_result, success_counter


def print_info(expected_values, outputs):
    print("PREDICTED == " + str([round(v) for v in outputs]))
    print("EXPECTED == " + str(expected_values))


def print_stats(dataset, data_type):
    print(data_type)
    print(dataset[data_type][Columns.LABEL.value].value_counts())
