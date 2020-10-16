import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import csv
import seaborn as sns
import mysql.connector
import scipy
from sklearn.linear_model import LinearRegression
from main import MyLinReg
from sklearn import svm
from sklearn.svm import SVR

sns.set_theme(style="ticks", color_codes=True)
conn = mysql.connector.Connect(host='localhost', user='rimak',password='Crima1993',database='carData_db')

#fname= input('Enter the  csv file name: ')
fname= 'carData.csv'
df= pd.read_csv(fname)

#print(df.columns)
df['Year'].astype(float)
df['Selling_Price'].astype(float)
df['Present_Price'].astype(float)
df['Kms_Driven'].astype(float)
print(df.describe())
#print('unique values of Owner: ',df['Owner'].unique())
#print('Sum of null values per column: \n', df.isnull().sum())

'''fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)
for i in range(3):
    for j in range(3):
        axs[i,j].hist(df[df.columns[3*i+j]])
        axs[i,j].set_xlabel(df.columns[3*i+j])
plt.show()
'''



cur= conn.cursor()
cur.execute('drop table if exists carData_table')
cur.execute('''
create table carData_table (Car_Name nvarchar(255), Year double, Selling_Price double, Present_Price double, Kms_Driven double, Fuel_Type nvarchar(255), Seller_Type nvarchar(255), Transmission nvarchar(255), Owner double)''')

    
for row in df.iterrows():
    testlist = row[1].values
    t= tuple(testlist)
    cur.execute(''' INSERT INTO carData_table(Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner) VALUES(%s, %s, %s, %s, %s, %s,%s, %s, %s)''', t)

    conn.commit()
#cur.execute("select * from carData_table")

#for x in cur:
  #print(x)

df1= pd.read_sql("select * from carData_table", conn);

 

pd.set_option('display.expand_frame_repr', False)

print(df1.columns)  
cur.close()
conn.close()

#sns.catplot(x="Year",y="Present_Price",data=df1)
#plt.show()

#numpy linear regression
fit = np.polyfit(df1['Year'].astype(float), df1['Selling_Price'].astype(float), 1)
print('Numpy linear regression parameters: ', fit)
poly = np.poly1d(fit)
prices_numpy= poly(df1['Year'].astype(float))
'''plt.plot(df1['Year'].astype(float), predicted_price)
plt.scatter(df1['Year'].astype(float), df1['Selling_Price'].astype(float), c='r')
plt.title('Numpy linear regression')
plt.show()
'''
#scipy linear regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df1['Year'].astype(float), df1['Selling_Price'].astype(float))
y_scipy= intercept + slope* df1['Year']
print('Scipy Linear regression parameters: ', slope, intercept, r_value, p_value, std_err)
'''plt.plot( df1['Year'].astype(float), df1['Selling_Price'].astype(float), 'o', label='original data')
plt.plot( df1['Year'].astype(float), intercept + slope* df1['Year'].astype(float), 'r', label='fitted line')
plt.legend()
plt.title('Scipy linear regression')
plt.show()
'''

#sklearn simple linear regression
years= df1['Year'].astype(float)
X= np.array([[year] for year in years])
print('X: ', X.shape)
Y= df1['Selling_Price'].astype(float)
reg = LinearRegression().fit(X, Y)
print('score linreg sklearn: ', reg.score(X, Y))
print(reg.coef_)
print(reg.intercept_)
prices_sklearn= reg.predict(X)
'''
plt.plot( df1['Year'].astype(float), df1['Selling_Price'].astype(float), 'o', label='original data')
plt.plot( df1['Year'].astype(float), intercept + slope* df1['Year'].astype(float), 'r', label='fitted line Scipy')
plt.plot( df1['Year'].astype(float),prices_sklearn, 'b', label='fitted line Sklearn')
plt.plot(df1['Year'].astype(float), prices_numpy, 'g', label= 'fitted line numpy')
#plt.ylim(0, 5)
plt.legend()
plt.title('Selling price vs years')
plt.show()
'''

#multivariate linear regression sklearn
print(df1['Transmission'].unique())
df1['Transmission'] = df1['Transmission'].replace({'Manual':'0', 'Automatic':'1'})
print(df1['Transmission'].unique())
df1['Transmission']= df1['Transmission'].astype(float)
X_MLR = df1[['Year','Kms_Driven', 'Transmission']] 
regr = LinearRegression()
regr.fit(X_MLR, Y)
y_sklearnMLR= regr.predict(X_MLR)
print('score Mlinreg sklearn: ', regr.score(X_MLR, Y))
print('Intercept MLR sklearn: \n', regr.intercept_)
print('Coefficients MLR sklearn : \n', regr.coef_)

#my regression model coded in main.py it is a class containing two functions simple_lin_reg et multiple_lin_reg
model= MyLinReg(X_MLR.values, Y)
print(model.simple_lin_reg())
print(model.multiple_lin_reg())

#svm
svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.1)
print('Go svr1')
svr_lin = SVR(kernel='linear')
print('Go svr2')
y_rbf = svr_rbf.fit(X_MLR, Y).predict(X_MLR)
print('score rbf: ', svr_rbf.score(X_MLR, Y) )
x_svr=df1['Year'].astype(float).values
x_svr_reshaped= x_svr.reshape(-1, 1)
y_lin = svr_lin.fit(x_svr_reshaped, Y).predict(x_svr_reshaped)
print('score lin: ',  svr_lin.score(x_svr_reshaped, Y))

plt.plot(X_MLR.values[:, 0], Y, '* k', label='data')
plt.plot(X_MLR.values[:, 0], y_rbf, '+ g', label='RBF model')
plt.plot(X_MLR.values[:, 0], y_lin, 'o r', label='Linear model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
#plt.show()


#output df
df_out = pd.DataFrame({'Year':x_svr , 'Selling_Price':Y , 'Y_np':prices_numpy,
                       'Y_Scipy':y_scipy, 'Y_sklearnSLR':prices_sklearn,
                       'Y_sklearnMLR':y_sklearnMLR, 
                       'Y_svr':y_lin, 'Y_svrMLR':y_rbf })
df_out.to_csv(r'out.csv')

