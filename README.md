### decision_tree
#### 通过决策树判断泰坦尼克号上乘客的生存状态。
### 流程
#### 1、读取数据
#### 2、选择有影响的特征，处理缺失值
#### 3、测试集和训练集划分
#### 4、特征工程，特征处理 --使用onehot编码处理字符型特征
#### 5、训练决策树
#### 6、预测结果
#### 7、导出决策树结构
### 问题
#### 1、使用python将dot文件转换为png文件
    pip 安装 pydot库
    import pydot
    (graph, ) = pydot.graph_from_dot_file('./tree.dot')
    graph.write_png('tree.png')
#### 2、windows python3库pydot运行出现：FileNotFoundError: [WinError 2] "dot" not found in path.（参考上传的pdf文件）
    1、安装软件：graphviz-2.38.msi  下载地址：https://graphviz.gitlab.io/_pages/Download/Download_windows.html
    2、配置环境变量
