import numpy as np

class MyLinReg():
    "MyLinReg"
    def __init__(self, x, y):
        self.x = x
        self.y= y
    
    def simple_lin_reg(self):
        x= self.x[:, 0]
        x_mean= np.mean(x)
        print('x_mean: ', x_mean)
        y_mean= np.mean(self.y)
        print('y_mean: ', y_mean)
        prod_xy= (x-x_mean)*(self.y-y_mean)
        sous= (x-x_mean)**2
        #print(sous)
        #print(sum(sous))
        a= (sum(prod_xy))/(sum(sous))
        b= y_mean - a*x_mean
        return a, b

    def multiple_lin_reg(self):
        x1= self.x
        new_column= np.array([[1.]*x1.shape[0]]).transpose()
        x= np.append(x1, new_column, axis=1)
        x.astype(float)
        xt= x.transpose()

        xtx= xt.dot(x)
        inv_xtx= np.linalg.inv(xt.dot(x))
        z= inv_xtx.dot(xt)
        return z.dot(self.y)
