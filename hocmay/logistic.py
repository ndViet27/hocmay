# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(2)

file_path = r"D:\hocmay\data2.xlsx"
data = pd.read_excel(file_path)
df = pd.DataFrame (data)
X1 = df.iloc[:, -2].values.tolist()
y = df.iloc[:, -1].values
X = np.array([X1])
 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)# thêm cột 1 vào đầu ma trận X
def sigmoid(s):
    return 1/(1 + np.exp(-s))

#Hàm huẩn luyện 
def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 52
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi #cập nhật w
            count += 1
            # Tiêu chí dừng
            if count%check_w_after == 0: # nếu biến đếm bằng đếm kiểm tra               
                if np.linalg.norm(w_new - w[-check_w_after]) < tol: #nếu chênh lệch giữa w mới và cũ nhỏ hơn 0.0001 thì trả về w 
                    return w
            w.append(w_new)
    return w
eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)

def logistic(x):#hàm giá trị đầu ra
    y = 0
    if (sigmoid(np.dot(w[-1].T,x))) >= 0.5:
        y=1
    else:
        y=0
    return y
# print(w[-1])
# for i in range(X.shape[1]):
#     print(logistic(X[:,i]))

# X0 = X[1, np.where(y == 0)][0]
# y0 = y[np.where(y == 0)]
# X1 = X[1, np.where(y == 1)][0]
# y1 = y[np.where(y == 1)]

# plt.plot(X0, y0, 'ro', markersize = 4)
# plt.plot(X1, y1, 'bs', markersize = 4)

# xx = np.linspace(0, 8, 1000)
# w0 = w[-1][0][0]
# w1 = w[-1][1][0]
# threshold = -w0/w1
# yy = sigmoid(w0 + w1*xx)
# plt.axis([0, 8, -1, 2])
# plt.plot(xx, yy, 'g-', linewidth = 2)
# plt.plot(threshold, .5, 'y^', markersize = 8)
# # plt.axvline(x=7.84, color='g') 
# plt.xlabel('Điểm quá trình')
# plt.ylabel('Dự đoán khả năng qua môn')
# plt.title('Đồ thị các điểm dữ liệu và đường dự đoán')
# plt.show()