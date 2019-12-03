def devide_dataset(data):
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
    validating_dataset = new_dataset(data, learning_rows_count + testing_rows_count,
                                     learning_rows_count + testing_rows_count + validating_rows_count)

    print("")
    print(learning_dataset.shape[0])
    print(testing_dataset.shape[0])
    print(validating_dataset.shape[0])

    return {
        "learning": learning_dataset,
        "testing": testing_dataset,
        "validating": validating_dataset
    }


def new_dataset(data, start_idx, end_idx):
    return data.iloc[start_idx: end_idx].reset_index(drop=True)


def find_dataset_indexes(all_rows_count):
    learning_rows_count = int(all_rows_count * 0.6)
    testing_rows_count = int(all_rows_count * 0.2)
    validating_rows_count = int(all_rows_count * 0.2)

    size = learning_rows_count + testing_rows_count + validating_rows_count
    correct = all_rows_count - size
    learning_rows_count = learning_rows_count + correct
    return learning_rows_count, testing_rows_count, validating_rows_count
