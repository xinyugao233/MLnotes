from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def decisioncls():
    """
    决策树进行乘客生存预测
    :return:
    """
    # 1、获取数据
    titan = pd.read_csv("titanic/titanic_train.csv")
    # 2、数据的处理
    x = titan[['Pclass', 'Age', 'Sex']]

    y = titan['Survived']

    # print(x , y)
    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x['Age'].fillna(x['Age'].mean(), inplace=False)
    #The inplace=True argument modifies the DataFrame directly.
    # If you want to avoid modifying the original DataFrame, set inplace=False.

    # 对于x转换成字典数据x.to_dict(orient="records")
    # [{"pclass": "1st", "age": 29.00, "sex": "female"}, {}]

    dict = DictVectorizer(sparse=False)

    x = dict.fit_transform(x.to_dict(orient="records"))

    # 分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


    rf = RandomForestClassifier()

    param = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5, 8, 15, 25, 30]}

    # 超参数调优
    gc = GridSearchCV(rf, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    print("随机森林预测的准确率为：", gc.score(x_test, y_test))

