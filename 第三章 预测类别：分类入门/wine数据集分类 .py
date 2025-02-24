# 使用wine数据库进行分类学习
from mlwpy.mlwpy import *

# 导入wine数据库并显示具体信息
wine = datasets.load_wine()

wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
display(wine_df)

# 划分训练集，测试集
(wine_train_ftrs, wine_test_ftrs,
 wine_train_tgt, wine_test_tgt) = skms.train_test_split(wine.data, wine.target, test_size=0.25)
print("Train features shape: ", wine_train_ftrs.shape)
print("Test features shape: ", wine_test_ftrs.shape)

# KNN分类器
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
fit = knn.fit(wine_train_ftrs, wine_train_tgt)
preds = fit.predict(wine_test_ftrs)

print(preds, wine_test_tgt, sep='\n')

# NB分类器
nb = naive_bayes.GaussianNB()
fit = nb.fit(wine_train_ftrs, wine_train_tgt)
preds = fit.predict(wine_test_ftrs)
print(wine_test_tgt, preds, sep='\n')
