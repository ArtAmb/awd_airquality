from second_stage import utils
from second_stage.labels import AirQuality
from second_stage.main import IMPORTANT_COLUMNS
from second_stage.processing import print_stats, start_process
from second_stage.utils import Watchstop, get_learning_data, get_testing_data, get_validating_data


def main():
    dataset = utils.read_dataset()
    print_stats(dataset, "learning")
    print_stats(dataset, "testing")
    print_stats(dataset, "validating")

    inputs_count = IMPORTANT_COLUMNS.__len__()
    outputs_count = AirQuality.__len__()

    print("PREPARING DATA")
    testing_inputs = get_testing_data(dataset, False)
    validating_inputs = get_validating_data(dataset, False)

    print("START TESTING")

    for idx in range(1, 7):
        neutral_network = utils.read_from_file(str(idx))
        epoch_result, success_counter = start_process(neutral_network, testing_inputs, False)
        epoch_result2, success_counter2 = start_process(neutral_network, validating_inputs, False)
        print("SIEC == " + str(idx))
        print("TESTY SUKCESY " + str(success_counter) + " / " + str(dataset["testing"].shape[0]))
        print("VALID SUKCESY " + str(success_counter2) + " / " + str(dataset["validating"].shape[0]))

if __name__ == '__main__':
    main()