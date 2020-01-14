import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import second_stage.dataset_devider as dd
from second_stage.dataset_manager import Columns
from second_stage.neural_network import NeutralNetwork, MOMENTUM_ACTIVE, MOMENTUM_RATE, LEARNING_RATE
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


def dumps_to_file(neutral_network, epoch_count=-1, learning_rows_count=-1, success_plot=[]):
    now = datetime.now()
    millis = int(round(time.time() * 1000))
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    objToSave = {
        "momentum": MOMENTUM_ACTIVE,
        "momentum_rate": MOMENTUM_RATE,
        "learning_rate": LEARNING_RATE,
        "epoch_count": epoch_count,
        "learning_rows_count": learning_rows_count,
        "SUCCESS_PLOT_DATA": success_plot,
        "neutral_network": neutral_network
    }

    jsonData = toJSON(objToSave)

    file = open("saved/" + dt_string + '.txt', 'w+')
    file.write(jsonData)

def read_from_file(file_name):
    file = open("saved/" + file_name + '.txt')
    jsonDATA = file.read()
    obj = json.loads(jsonDATA)
    nn = NeutralNetwork(0, [])
    nn.load(obj["neutral_network"])

    momentum = obj["momentum"]
    if momentum:
        print("Propagacja wsteczna z MOMENTUM")
        print("Wspołczynnik momentum: " + str(obj["momentum_rate"]))
    else:
        print("Propagacja wsteczna wersja standardowa")
    print("Współczynnik uczenia: " + str(obj["learning_rate"]))
    print("Ilosc epok: " + str(obj["epoch_count"]))


    return nn


def read_dataset():
    print("Air quality in Changping, one of the Pekin's district, in 2013.03.01 - 2017.02.28")
    data = pd.read_csv("labeled_dataset.csv", delimiter=",")
    return dd.devide_dataset(data)


class Normalizer:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.max_minus_min = max - min

    def normalize(self, value):
        return (value - self.min) / self.max_minus_min


def get_learning_data(dataset, shuffle):
    return get_data(dataset, "learning", shuffle)

def get_testing_data(dataset, shuffle):
    return get_data(dataset, "testing", shuffle)

def get_validating_data(dataset, shuffle):
    return get_data(dataset, "validating", shuffle)

def get_data(dataset, type, shuffle):
    if shuffle:
        rows = dataset[type].sample(frac=1)
    else:
        rows = dataset[type]

    MIN_MAX_NORMALIZER = {}
    for col in IMPORTANT_COLUMNS:
        tmp_col = rows[col.value]
        MIN_MAX_NORMALIZER[col.value] = Normalizer(min(tmp_col), max(tmp_col))

    result = []
    for index, row in rows.iterrows():
        result.append({
            "input": prepare_inputs(row, MIN_MAX_NORMALIZER),
            "label": row[Columns.LABEL.value]})

    return result


def prepare_inputs(row, MIN_MAX_NORMALIZER):
    inputs = []

    for col in IMPORTANT_COLUMNS:
        inputs.append(MIN_MAX_NORMALIZER[col.value].normalize(row[col.value]))
    return inputs
