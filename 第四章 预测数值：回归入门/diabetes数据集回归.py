from mlwpy.mlwpy import *

# 加载diabetes数据集
diabetes = datasets.load_diabetes()

# 划分数据集
tts = skms.train_test_split(diabetes.data, diabetes.target, test_size=0.25)

(diabetes_train_ftrs, diabetes_test_ftrs,
 diabetes_train_tgt, diabetes_test_tgt) = tts

diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target
display(diabetes_df)

# KNN回归器
knn = neighbors.KNeighborsRegressor(n_neighbors=3)
fit = knn.fit(diabetes_train_ftrs, diabetes_train_tgt)
preds = fit.predict(diabetes_test_ftrs)

print(preds, diabetes_test_tgt, sep='\n')

error = metrics.mean_squared_error(diabetes_test_tgt, preds)
print(error)