# 使用breast_cancer数据库进行分类学习
from mlwpy.mlwpy import *

# 导入breast_cancer数据库并显示具体信息
breast_cancer = datasets.load_breast_cancer()

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target
display(breast_cancer_df)

# 划分训练集，测试集
(breast_cancer_train_ftrs, breast_cancer_test_ftrs,
 breast_cancer_train_tgt, breast_cancer_test_tgt) = skms.train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.25)
print("Train features shape: ", breast_cancer_train_ftrs.shape)
print("Test features shape: ", breast_cancer_test_ftrs.shape)

# KNN分类器
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
fit = knn.fit(breast_cancer_train_ftrs, breast_cancer_train_tgt)
preds = fit.predict(breast_cancer_test_ftrs)

print(preds, breast_cancer_test_tgt, sep='\n')

# NB分类器
nb = naive_bayes.GaussianNB()
fit = nb.fit(breast_cancer_train_ftrs, breast_cancer_train_tgt)
preds = fit.predict(breast_cancer_test_ftrs)
print(breast_cancer_test_tgt, preds, sep='\n')
