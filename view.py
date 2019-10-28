import matplotlib.pyplot as plt

def print_title(title):
    print()
    print("##################################")
    print(title)
    print("##################################")

def showScatterGraph(data, col1, col2):
    data.plot.scatter(col1, col2, title="Scatter plot graph\n" + col1 + " x " + col2)
    plt.show()