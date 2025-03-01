import numpy as np
np.random.seed(42)
from sklearn import(datasets, neighbors, model_selection as skms, linear_model, metrics)

diabetes = datasets.load_diabetes()
tts = skms.train_test_split(diabetes.data, diabetes.target, test_size=0.25)
(diabetes_train, diabetes_test, diabetes_train_tgt, diabetes_test_tgt) = tts

models = {'KNN': neighbors.KNeighborsRegressor(n_neighbors=3), 'Linear' : linear_model.LinearRegression()}

for name, model in models.items():
    fit = model.fit(diabetes_train, diabetes_train_tgt)
    preds = fit.predict(diabetes_test)

    score = np.sqrt(metrics.mean_squared_error(diabetes_test_tgt, preds))
    print("{:>6s} : {:0.2f}".format(name, score))