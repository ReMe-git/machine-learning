from sklearn import (datasets, metrics, model_selection as skms, naive_bayes, neighbors)

# 导入iris数据库
iris = datasets.load_iris()
# 分割训练和测试数据
(iris_train_ftrs, iris_test_ftrs,
 iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.90,
                                                        random_state=42)

# KNN,NB分类器
models = {'KNN' : neighbors.KNeighborsClassifier(n_neighbors=3),
          'NB' : naive_bayes.GaussianNB()}
for name, model in models.items():
    fit = model.fit(iris_train_ftrs, iris_train_tgt)
    predictions = fit.predict(iris_test_ftrs)

    score = metrics.accuracy_score(iris_test_tgt, predictions)
    print("{:>3s}: {:0.2f}".format(name, score))