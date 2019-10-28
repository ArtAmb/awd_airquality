from abc import abstractmethod

import pandas as pd


class ColumnValidator:
    @abstractmethod
    def valid(self, value): raise NotImplementedError


class TimeValidator(ColumnValidator):
    def valid(self, value):
        return True


class DateValidator(ColumnValidator):
    def valid(self, value):
        return True


class NoNegativeValueValidator(ColumnValidator):
    def valid(self, value):

        if self.toNum(value) < 0:
            return False

        return True

    def toNum(self, value):
        if isinstance(value, str):
            return pd.to_numeric(value.replace(",", "."))
        else:
            return value


class RowValidator:
    column_names_to_process = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)"]
    all_column_names = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)",
                        "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
    column_names = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)",
                    "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
    validators = dict({"Date": DateValidator(),
                       "Time": TimeValidator(),
                       "CO(GT)": NoNegativeValueValidator(),
                       "PT08.S1(CO)": NoNegativeValueValidator(),
                       "NMHC(GT)": NoNegativeValueValidator(),
                       "C6H6(GT)": NoNegativeValueValidator(),
                       "PT08.S2(NMHC)": NoNegativeValueValidator(),
                       "NOx(GT)": NoNegativeValueValidator(),
                       "PT08.S3(NOx)": NoNegativeValueValidator(),
                       "NO2(GT)": NoNegativeValueValidator(),
                       "PT08.S4(NO2)": NoNegativeValueValidator(),
                       "PT08.S5(O3)": NoNegativeValueValidator(),
                       "T": NoNegativeValueValidator(),
                       "RH": NoNegativeValueValidator(),
                       "AH": NoNegativeValueValidator()}
                      )

    def __init__(self, column_names):
        self.column_names = column_names

        for colName in self.column_names:
            if self.validators[colName] == None:
                raise Exception("There is no validator for column: " + colName)

    def find_first_error_column(self, row):
        for col_name in self.column_names:
            isOK = self.validators[col_name].valid(row[col_name])
            if not isOK:
                return col_name

        return None

    def getValues(self, data, column_name):
        result = pd.DataFrame()
        for cell in data[column_name]:
            result.append(cell)
