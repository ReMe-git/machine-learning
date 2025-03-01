from mlwpy import *
from ucimlrepo import fetch_ucirepo

# 加载heart_disease数据集
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
print(X, y, sep='\n')

features = np.array(X)
features = np.nan_to_num(features, nan=0)
target = np.array(y).reshape(303)

(heart_disease_train_ftrs, heart_disease_test_ftrs,
 heart_disease_train_tgt, heart_disease_test_tgt) = skms.train_test_split(features, target, test_size=0.90, random_state=42)

# KNN分类器
knn = neighbors.KNeighborsClassifier(n_neighbors=4)
fit = knn.fit(heart_disease_train_ftrs, heart_disease_train_tgt)
preds = fit.predict(heart_disease_test_ftrs)

print(preds, heart_disease_test_tgt, sep='\n')

# NB分类器
nb = naive_bayes.GaussianNB()
fit = nb.fit(heart_disease_train_ftrs, heart_disease_train_tgt)
preds = fit.predict(heart_disease_test_ftrs)
print(heart_disease_test_tgt, preds, sep='\n')
