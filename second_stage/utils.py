import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import second_stage.dataset_devider as dd
from second_stage.dataset_manager import Columns
from second_stage.neural_network import NeutralNetwork
from second_stage.processing import IMPORTANT_COLUMNS


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
        print(desc + " " + str(self.duration()) + ' ms')
        self.start()


def convert(o):
    if isinstance(o, np.ndarray):
        return o.tolist()

    return o.__dict__


def toJSON(obj):
    return json.dumps(obj, default=lambda o: convert(o), sort_keys=True, indent=4)


def dumps_to_file(neutral_network):
    now = datetime.now()
    millis = int(round(time.time() * 1000))
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    jsonData = toJSON(neutral_network)
    file = open("saved/" + dt_string + '.txt', 'w+')
    file.write(jsonData)


def read_from_file(file_name):
    file = open("saved/" + file_name + '.txt')
    jsonDATA = file.read()
    nn = NeutralNetwork(0, [])
    nn.load(jsonDATA)
    return nn


def read_dataset():
    print("Air quality in Changping, one of the Pekin's district, in 2013.03.01 - 2017.02.28")
    data = pd.read_csv("labeled_dataset.csv", delimiter=",")
    return dd.devide_dataset(data)

def get_learning_data(dataset, shuffle):
    if shuffle:
        rows = dataset["learning"].sample(frac=1)
    else:
        rows = dataset["learning"]

    result = []
    for index, row in rows.iterrows():
        result.append({
            "input": prepare_inputs(row),
            "label": row[Columns.LABEL.value]})

    return result

def prepare_inputs(row):
    inputs = []
    for col in IMPORTANT_COLUMNS:
        inputs.append(row[col.value])
    return inputs