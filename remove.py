def remove_empty_cells(data):
    data = data.dropna(how="all", axis='index')
    data = data.dropna(how="all", axis='columns')
    data = data.dropna(axis='index')
    return data.dropna(axis='columns')


def findOkElements(elements):
    result = []
    for el in elements:
        if el > 0:
            result.append(el)
    return result