from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge

def myRDregression():
    """
    岭回归预测房子价格
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

    # Optional: You could scale y (target) if required, but it's usually not necessary
    # Scaling target y is generally unnecessary for Ridge regression
    std_y = StandardScaler()
    y_train = y_train.reshape(-1, 1)  # Reshape to 2D for scaling
    y_test = y_test.reshape(-1, 1)    # Reshape to 2D for scaling
    y_train = std_y.fit_transform(y_train)  # Standardize target
    y_test = std_y.transform(y_test)       # Apply same transformation to test set

    # Initialize and fit the Ridge regression model
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)

    # Print the coefficients
    print("岭回归的权重参数为：", rd.coef_)

    # Predict with the model and inverse transform the predictions
    y_rd_predict = rd.predict(x_test)
    y_rd_predict = std_y.inverse_transform(y_rd_predict.reshape(-1, 1))  # Inverse transform predictions

    # Print the predictions
    print("岭回归的预测的结果为：", y_rd_predict)

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, y_rd_predict)
    print("岭回归的均方误差为：", mse)

    return None

# Call the function to run the Ridge regression
myRDregression()

if __name__ == '__main__':
    myRDregression()