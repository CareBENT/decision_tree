import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot


"""
# pandas DataFrame 显示设置
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
"""


def decision_tree_titanic():
    """
    泰坦尼克号数据：在泰坦尼克号和titanic2数据帧描述泰坦尼克号上的个别乘客的生存状态。
    通过决策树判断泰坦尼克号上乘客的生存状态。
    :return:
    """
    # 1、读取数据
    titan_data = pd.read_csv("./data/http___biostat.mc.va_20190728_220710.txt")

    # 2、选择有影响的特征，处理缺失值
    # -a) 处理数据，找出特征值和目标值
    x = titan_data.loc[:, ['pclass', 'age', 'sex']]
    y = titan_data['survived']

    # -b) 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 3、测试集和训练集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 4、特征工程，特征处理 --使用onehot编码处理字符型特征
    dict = DictVectorizer()
    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    x_test = dict.transform(x_test.to_dict(orient='records'))

    print(dict.get_feature_names())

    # 5、训练决策树
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)

    # 6、预测结果
    print("预测准确率：", tree.score(x_test, y_test))

    # 7、导出决策树结构
    export_graphviz(tree, out_file='./tree.dot', feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])
    (graph, ) = pydot.graph_from_dot_file('./tree.dot')
    graph.write_png('tree.png')

    return None


if __name__ == "__main__":
    decision_tree_titanic()
