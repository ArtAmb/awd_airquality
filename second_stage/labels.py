from second_stage.dataset_manager import Columns
import statistics as stats
from enum import Enum

MIN_VALUE = -9999999
MAX_VALUE = 9999999


class QualityRange:

    def __init__(self, label, start, stop):
        self.label = label
        self.start = start
        self.stop = stop

    def isInRange(self, value):
        if value >= self.start and value < self.stop:
            return True
        else:
            return False

    def getLabel(self):
        return self.label


class AirQuality(Enum):
    EXCELLENT = 6
    GOOD = 5
    LIGHTLY_POLLUTED = 4
    MODERATELY_POLLUTED = 3
    HEAVILY_POLLUTED = 2
    SEVERELY_POLLUTED = 1
    UNDEFINED = 0


def select_label(data_row):
    data_row(Columns.PM10)


# No: row number
# year: year of data in this row
# month: month of data in this row
# day: day of data in this row
# hour: hour of data in this row
# PM2.5: PM2.5 concentration (ug/m^3)
# PM10: PM10 concentration (ug/m^3)
# SO2: SO2 concentration (ug/m^3)
# NO2: NO2 concentration (ug/m^3)
# CO: CO concentration (ug/m^3)
# O3: O3 concentration (ug/m^3)
# TEMP: temperature (degree Celsius)
# PRES: pressure (hPa)
# DEWP: dew point temperature (degree Celsius)
# RAIN: precipitation (mm)
# wd: wind direction
# WSPM: wind speed (m/s)
# station: name of the air-quality monitoring site

class Labels:
    labels_ranges = {
        Columns.PM10: [
            QualityRange(AirQuality.EXCELLENT, MIN_VALUE, 20),
            QualityRange(AirQuality.GOOD, 20, 50),
            QualityRange(AirQuality.LIGHTLY_POLLUTED, 50, 80),
            QualityRange(AirQuality.MODERATELY_POLLUTED, 80, 110),
            QualityRange(AirQuality.HEAVILY_POLLUTED, 110, 150),
            QualityRange(AirQuality.SEVERELY_POLLUTED, 150, MAX_VALUE)],

        Columns.PM25: [
            QualityRange(AirQuality.EXCELLENT, MIN_VALUE, 13),
            QualityRange(AirQuality.GOOD, 13, 35),
            QualityRange(AirQuality.LIGHTLY_POLLUTED, 35, 55),
            QualityRange(AirQuality.MODERATELY_POLLUTED, 55, 75),
            QualityRange(AirQuality.HEAVILY_POLLUTED, 75, 110),
            QualityRange(AirQuality.SEVERELY_POLLUTED, 110, MAX_VALUE)],

        Columns.O3: [
            QualityRange(AirQuality.EXCELLENT, MIN_VALUE, 70),
            QualityRange(AirQuality.GOOD, 70, 120),
            QualityRange(AirQuality.LIGHTLY_POLLUTED, 120, 150),
            QualityRange(AirQuality.MODERATELY_POLLUTED, 150, 180),
            QualityRange(AirQuality.HEAVILY_POLLUTED, 180, 240),
            QualityRange(AirQuality.SEVERELY_POLLUTED, 240, MAX_VALUE)],

        Columns.NO2: [
            QualityRange(AirQuality.EXCELLENT, MIN_VALUE, 40),
            QualityRange(AirQuality.GOOD, 40, 100),
            QualityRange(AirQuality.LIGHTLY_POLLUTED, 100, 150),
            QualityRange(AirQuality.MODERATELY_POLLUTED, 150, 200),
            QualityRange(AirQuality.HEAVILY_POLLUTED, 200, 400),
            QualityRange(AirQuality.SEVERELY_POLLUTED, 400, MAX_VALUE)],

        Columns.SO2: [
            QualityRange(AirQuality.EXCELLENT, MIN_VALUE, 50),
            QualityRange(AirQuality.GOOD, 50, 100),
            QualityRange(AirQuality.LIGHTLY_POLLUTED, 100, 200),
            QualityRange(AirQuality.MODERATELY_POLLUTED, 200, 350),
            QualityRange(AirQuality.HEAVILY_POLLUTED, 350, 500),
            QualityRange(AirQuality.SEVERELY_POLLUTED, 500, MAX_VALUE)],

        Columns.CO: [
            QualityRange(AirQuality.EXCELLENT, MIN_VALUE, 3000),
            QualityRange(AirQuality.GOOD, 3000, 7000),
            QualityRange(AirQuality.LIGHTLY_POLLUTED, 7000, 11000),
            QualityRange(AirQuality.MODERATELY_POLLUTED, 11000, 15000),
            QualityRange(AirQuality.HEAVILY_POLLUTED, 15000, 21000),
            QualityRange(AirQuality.SEVERELY_POLLUTED, 21000, MAX_VALUE)]

    }


def classify(attribute, value):
    ranges = Labels.labels_ranges[attribute]

    for range in ranges:
        if range.isInRange(value):
            return range.getLabel()

    return AirQuality.UNDEFINED


DECISIVE_COLUMNS = [Columns.PM10, Columns.PM25, Columns.O3, Columns.NO2, Columns.SO2, Columns.CO]

def classify_row(row):
    labels = []
    for col in DECISIVE_COLUMNS:
        label = classify(col, row[col.value])
        if AirQuality.UNDEFINED == label:
            print("UNDEFINED No == " + str(row[Columns.NO]))
        labels.append(label.value)

    return AirQuality(round(stats.mean(labels)))

