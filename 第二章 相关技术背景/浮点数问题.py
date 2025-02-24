# float类型不能直接比较大小，需要使用numpy的allclose()函数比较
import numpy as np

print(np.allclose(1.1 + 2.2, 3.3))