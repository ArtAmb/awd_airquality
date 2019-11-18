import pandas as pd


def clean(data):
    data.dropna(how="all", axis='index', inplace=True)
    data.dropna(how="all", axis='columns', inplace=True)
    data.fillna(inplace=True, method="ffill")
    data.fillna(inplace=True, method="bfill")


def main():
    print("Air quality in Changping, one of the Pekin's district, in 2013.03.01 - 2017.02.28")
    data = pd.read_csv("dataset.csv", delimiter=",")

    print(data.mean())
    clean(data)
    print(data.mean())


    all_rows_count = data.shape[0]

    learning_rows_count, testing_rows_count, validating_rows_count = find_dataset_indexes(all_rows_count)

    print("")
    print(learning_rows_count)
    print(testing_rows_count)
    print(validating_rows_count)

    print("")
    print(all_rows_count)
    print(learning_rows_count + testing_rows_count + validating_rows_count)

    learning_dataset = new_dataset(data, 0, learning_rows_count)
    testing_dataset = new_dataset(data, learning_rows_count, learning_rows_count + testing_rows_count)
    validating_dataset = new_dataset(data, learning_rows_count + testing_rows_count, learning_rows_count + testing_rows_count + validating_rows_count)

    print("")
    print(learning_dataset.shape[0])
    print(testing_dataset.shape[0])
    print(validating_dataset.shape[0])


def new_dataset(data, start_idx,  end_idx):
    return data.iloc[start_idx: end_idx].reset_index(drop=True)


def find_dataset_indexes(all_rows_count):
    learning_rows_count = int(all_rows_count * 0.6)
    testing_rows_count = int(all_rows_count * 0.2)
    validating_rows_count = int(all_rows_count * 0.2)

    size = learning_rows_count + testing_rows_count + validating_rows_count
    correct = all_rows_count - size
    learning_rows_count = learning_rows_count + correct
    return learning_rows_count, testing_rows_count, validating_rows_count


if __name__ == '__main__':
    main()
