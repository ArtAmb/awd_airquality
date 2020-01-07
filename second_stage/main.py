import pandas as pd

import second_stage.dataset_devider as dd
from second_stage import utils
from second_stage.labels import AirQuality
from second_stage.neural_network import NeutralNetwork
from second_stage.processing import print_stats, IMPORTANT_COLUMNS, start_process, show_not_important_errors, \
    show_important_errors, show_success_plot
from second_stage.utils import Watchstop, get_learning_data


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
    learning_inputs = get_learning_data(dataset, True)
    print("START LEARNING")
    ws = Watchstop()
    for epoch in range(0, 10):
        ws.start()
        epoch_result, success_counter = start_process(neutral_network, learning_inputs, True)
        print("EPOKA " + str(epoch) + " SUKCESY " + str(success_counter) + " / " + str(dataset["learning"].shape[0]))
        # epoch_result2, success_counter2 = start_process(neutral_network, learning_inputs, False)
        # print("EPOKA " + str(epoch) + " SUKCESY " + str(success_counter2) + " / " + str(dataset["learning"].shape[0]))
        epoch_result2, success_counter2 = start_process(neutral_network, learning_inputs, False)
        print("TESTY SUKCESY " + str(success_counter2) + " / " + str(dataset["learning"].shape[0]))
        ws.stop_print_start("FULL TIME ==")
        # print(epoch_result)

    # show_important_errors()
    # show_not_important_errors()
    # show_success_plot()
    # show_errors_plot()


    print("STOP LEARNING")

    print("START TESTING")
    # utils.dumps_to_file(neutral_network)
    # neutral_network = utils.read_from_file("06_01_2020__20_02_21")
    ws.start()
    epoch_result, success_counter = start_process(neutral_network, learning_inputs, False)
    print("TESTY SUKCESY " + str(success_counter) + " / " + str(dataset["learning"].shape[0]))
    ws.stop_print_start("FULL TIME ==")


if __name__ == '__main__':
    main()
