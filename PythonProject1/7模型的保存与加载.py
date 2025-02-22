# 使用线性模型进行预测
# 使用正规方程求解
import joblib
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 此时在干什么？
lr.fit(x_train, y_train)
# 保存训练完结束的模型
joblib.dump(lr, "test.pkl")



# 通过已有的模型去预测房价
model = joblib.load("test.pkl")
print("从文件加载进来的模型预测房价的结果：", std_y.inverse_transform(model.predict(x_test)))