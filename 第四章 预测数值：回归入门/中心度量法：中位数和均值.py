from mlwpy import *
# 为了从多个近邻样例中获取测试样例target的回归值，有多种方法，这里使用了中位数，均值与加权平均值

value = np.array([1, 7, 11])
# 中位数：邻近样例目标值的中位数， 当数据两端发生变化时中位数不变，具有很好的鲁棒性
print(np.mean(value))

# 均值： 满足sum(distance(s, mean), for smaller) = sum(distance(s, mean), for bigger),平衡了左右两边值的总距离， mean = sum(d) / len(d)
print(np.median(value))

# 加权平均值：weight = (1 / distances) / sum(1 / distances)
distances = np.array([2.0, 4.0, 4.0])
closeness = 1.0 / distances
weights = closeness / np.sum(closeness)

print(np.dot(value, weights))