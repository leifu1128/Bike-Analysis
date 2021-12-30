import numpy as np
import pandas as pd


def main():
    # import dataset and data
    bridgeList = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']
    wthrList = ['High Temp (°F)', 'Low Temp (°F)', 'Precipitation', 'Total']
    bridge_data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv', usecols=bridgeList, quoting=2)
    wthr_data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv', usecols=wthrList)

    # parse bridge data
    bridge_data = bridge_data.to_numpy()
    bridge_data = np.frompyfunc(lambda x: x.replace(',', ''), 1, 1)(bridge_data).astype(np.single)

    # split into X and Y
    X_bridge_data = bridge_data[:, 0:4]
    Y_bridge_data = bridge_data[:, 4]

    # calculate correlation coefficient
    print("Question 1 Results:")
    for i in range(4):
        print(np.corrcoef(X_bridge_data[:, i], y=Y_bridge_data))

    # calculate mse
    mse = MSE(X_bridge_data, Y_bridge_data)
    print(mse)

    # parse weather data
    wthr_data = wthr_data.to_numpy()
    wthr_data = np.delete(wthr_data, 3, 0)
    wthr_data[:, 3] = np.frompyfunc(lambda x: x.replace(',', ''), 1, 1)(wthr_data[:, 3])
    wthr_data[:, 2:4] = np.frompyfunc(lambda x: x.replace('T', '0'), 1, 1)(wthr_data[:, 2:4])
    wthr_data = wthr_data.astype(np.single)
    per_data = wthr_data[:, 2]
    wthr_data = np.hstack([np.ones([wthr_data.shape[0], 1], dtype=np.single), wthr_data])

    # calculate least squares
    b1 = least_squares(wthr_data[:, 0:4], wthr_data[:, 4])  # without precipitation
    b2 = least_squares(wthr_data[:, 0:3], wthr_data[:, 4])  # without precipitation

    # test each coefficient
    z_scores1 = test_coeff(b1, wthr_data[:, 0:4], wthr_data[:, 4])
    z_scores2 = test_coeff(b2, wthr_data[:, 0:3], wthr_data[:, 4])
    print("Question 2 Results:")
    print(z_scores1)
    print(z_scores2)

    # parse bridge data
    per_data = per_data.astype(np.single)
    bridge_data2 = np.hstack([np.ones([bridge_data.shape[0], 1], dtype=np.single), bridge_data])
    bridge_data2 = np.delete(bridge_data2, 3, 0)

    # calculate least squares
    b3 = least_squares(bridge_data2, per_data)
    b4 = least_squares(bridge_data2[:, 0:5], per_data)

    # test each coefficient
    z_scores3 = test_coeff(b3, bridge_data2, per_data)
    z_scores4 = test_coeff(b4, bridge_data2[:, 0:5], per_data)
    print("Question 3 Results:")
    print(z_scores3)
    print(z_scores4)

    return


def least_squares(X, y):
    B = np.linalg.inv(X.T @ X) @ (X.T @ y)

    return B


def MSE(X, y):
    mse = []

    for i in range(np.size(X, 1)):
        x_temp = np.delete(X, i, 1)
        x_temp = np.hstack([np.ones([x_temp.shape[0], 1], dtype=np.single), x_temp])

        b = least_squares(x_temp, y)
        y_pred = b * x_temp
        y_pred = y_pred.sum(axis=1)
        MSE = np.square((y - y_pred))
        MSE = np.sum(MSE) / len(y)
        mse.append(MSE)

    return mse


def test_coeff(b, X, y):
    zscores = []

    # calculate predicted values
    y_pred = b * X
    y_pred = y_pred.sum(axis=1)

    # hypothesis test
    for i in range(1, np.size(X, 1)):
        x = X[:, i]

        # denominator of SE
        SED = np.square((x - np.mean(x)))
        SED = np.sum(SED)**(1/2)

        # numerator of SE
        SEN = np.square((y - y_pred))
        SEN = (np.sum(SEN) / (len(y) - 2))**(1/2)

        SE = SEN / SED
        zscores.append(b[i] / SE)

    return zscores


if __name__ == '__main__':
    main()
