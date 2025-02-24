# numpy的数组点积规则与矩阵乘法类似
# 数组形状需要满足要求及左边数组行数与右边数组列数相同

import numpy as np

# 简单示例
oned_vec= np.arange(5)
print(oned_vec, "-->", oned_vec * oned_vec)
print("self dot:", np.dot(oned_vec, oned_vec))

# 使用一行一列的示例
row_vec= np.arange(5).reshape(1,5)
col_vec = np.arange(0, 50, 10).reshape(5,1)

print("row_vec:", row_vec, row_vec.shape,
      "col_vec:", col_vec, col_vec.shape,
      "dot:",np.dot(row_vec, col_vec), sep='\n')

# 交换次序
print(np.dot(col_vec, row_vec))

# 一维数组与二维数组

print(np.dot(oned_vec, col_vec))

# 次序反转则出现形状错误
try:
    np.dot(col_vec, oned_vec)
except ValueError as e:
    print("Failed: ", e)

def rdot(arr, brr):
    return np.dot(brr, arr)

# 使用rdot来解决
print(rdot(col_vec, oned_vec))


# 使用rdot来与实际数学公式保持一致
# 如 y=wD
D = np.array([[1, 3],
              [2, 5],
              [2, 7],
              [3, 2]])

w = np.array([1.5, 2.5])

print(np.dot(D, w), rdot(w, D), sep='\n')