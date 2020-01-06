from second_stage import utils
from second_stage.labels import AirQuality
from second_stage.main import IMPORTANT_COLUMNS
from second_stage.processing import print_stats, start_process
from second_stage.utils import Watchstop, get_learning_data


def main():
    dataset = utils.read_dataset()
    print_stats(dataset, "learning")
    print_stats(dataset, "testing")
    print_stats(dataset, "validating")

    inputs_count = IMPORTANT_COLUMNS.__len__()
    outputs_count = AirQuality.__len__()

    print("PREPARING DATA")
    learning_inputs = get_learning_data(dataset, False)

    print("START TESTING")
    neutral_network = utils.read_from_file("06_01_2020__22_40_06")
    epoch_result, success_counter = start_process(neutral_network, learning_inputs, False)
    print("TESTY SUKCESY " + str(success_counter) + " / " + str(dataset["learning"].shape[0]))

if __name__ == '__main__':
    main()