from timeit import timeit

from second_stage.main import IMPORTANT_COLUMNS


def main():
    tmp = [10] * 5

    test1 = """
inputs = []
for i in range(0, 7):
     inputs.append(10)
 """

    test2 =  """
inputs = []
for i in range(0, 7):
     inputs += [10]
 """

    print(timeit(test1))
    print(timeit(test2))

    # res = np.multiply(np.array([2,2,8]), 0.5 * 0.5)
    # print(res)
    #
    # res = np.array([[1, 2, 3],
    #                 [4, 5, 6],
    #                 [7, 8, 9]])
    # print(res)
    # # plt.plot([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # res = res.append([10, 11, 12])
    # print(res)
    # plt.plot(res)
    # plt.show()


if __name__ == '__main__':
    main()
