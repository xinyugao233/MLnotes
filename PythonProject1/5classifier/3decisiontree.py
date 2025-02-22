import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

    print(dict.get_feature_names_out())
    print(x)

    # 分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 进行决策树的建立和预测
    dc = DecisionTreeClassifier(max_depth=5)

    dc.fit(x_train, y_train)

    print("预测的准确率为：", dc.score(x_test, y_test))

    return None
if __name__ == '__main__':
    decisioncls()

