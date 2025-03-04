from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor

def mylinearregression():
    """
    线性回归预测房子价格
    :return:
    """
    # Load the dataset
    housing = fetch_california_housing()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.3, random_state=24)

    # Standardize the features (X)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # Standardize the target (y) only if needed
    std_y = StandardScaler()
    y_train = y_train.reshape(-1, 1)  # Reshape to 2D for scaling
    y_test = y_test.reshape(-1, 1)    # Reshape to 2D for scaling

    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    # Train the linear regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Make predictions
    y_lr_predict = lr.predict(x_test)

    # Inverse transform the predictions to original scale
    y_lr_predict = std_y.inverse_transform(y_lr_predict.reshape(-1, 1))

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, y_lr_predict)
    print(f"正规方程的均方误差为：{mse}")

    # Display the coefficients
    print(f"Model Coefficients: {lr.coef_}")

    return None


def mySGDregression():
    """
    线性回归预测房子价格
    :return:
    """
    # Load the dataset
    housing = fetch_california_housing()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.3, random_state=24)

    # Standardize the features (X)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # Standardize the target (y) only if needed
    std_y = StandardScaler()
    y_train = y_train.reshape(-1, 1)  # Reshape to 2D for scaling
    y_test = y_test.reshape(-1, 1)    # Reshape to 2D for scaling

    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    # Train the linear regression model
    sr = SGDRegressor()
    sr.fit(x_train, y_train)

    # Make predictions
    y_sr_predict = sr.predict(x_test)

    # Inverse transform the predictions to original scale
    y_sr_predict = std_y.inverse_transform(y_sr_predict.reshape(-1, 1))

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, y_sr_predict)
    print(f"正规方程的均方误差为：{mse}")

    # Display the coefficients
    print(f"Model Coefficients: {sr.coef_}")

    return None

if __name__ == '__main__':
    mySGDregression()