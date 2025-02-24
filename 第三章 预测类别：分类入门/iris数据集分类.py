# 使用iris数据库进行分类学习
from mlwpy.mlwpy import *

# 导入iris数据库并显示具体信息
iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
display(iris_df)

# 划分训练集，测试集
(iris_train_ftrs, iris_test_ftrs,
 iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data, iris.target, test_size=0.25)
print("Train features shape: ", iris_train_ftrs.shape)
print("Test features shape: ", iris_test_ftrs.shape)

# KNN分类器
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
fit = knn.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)

print(preds, iris_test_tgt, sep='\n')

# NB分类器
nb = naive_bayes.GaussianNB()
fit = nb.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)
print(iris_test_tgt, preds, sep='\n')
