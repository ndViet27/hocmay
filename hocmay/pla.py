import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
np.random.seed(2)

file_path = r"D:\hocmay\data1.xlsx"
data = pd.read_excel(file_path)
df = pd.DataFrame (data)
X0 = df.iloc[:, 0].values
X1 = df.iloc[:, 1].values
y = df.iloc[:,-1].values
X = np.vstack((X0,X1))
y = np.reshape(y,(1,108))
# Xbar 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)


def h(w, x):#hàm lấy dấu của w*x    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):#hàm kiểm tra dầu của w*x với nhãn y
    return np.array_equal(h(w, X), y) 

#hàm huấn luyện
def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi) != yi: # nếu nhãn của w*x khác nhãn y thì cập nhật w
                w_new = w[-1] + yi*xi 
                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w)

d = X.shape[0]
w_init = np.random.randn(d, 1)
w = perceptron(X, y, w_init)

def perceptron(x):# hàm giá trị đầu ra
    y=0
    if (h(w[-1],x)) == 1:
        y=1
    else:
        y=0
    return y

# print(X[:,0])
# for i in range(X.shape[1]):
#     print(perceptron(X[:,i]))
# print(w[-1])

# # Vẽ các điểm dữ liệu
# plt.figure(figsize = (10, 6))
# plt.scatter(X0, X1, c = y[0], cmap = 'coolwarm')

# # Tính toán và vẽ đường biên
# x_axis = np.linspace(np.min(X0), np.max(X0), 100)
# y_axis = -(w[-1][0] + w[-1][1]*x_axis)/w[-1][2]
# plt.plot(x_axis, y_axis)

# plt.xlabel('LT')
# plt.ylabel('TH')
# plt.title('Đồ thị các điểm dữ liệu và đường biên')
# plt.show()