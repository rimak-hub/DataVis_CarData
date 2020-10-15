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
print('Hi its rima:::')
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
#print(df.describe(include= "all"))
#print('unique values of Owner: ',df['Owner'].unique())

#print('Sum of null values per column: \n', df.isnull().sum())
'''fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)
for i in range(3):
    for j in range(3):
        axs[i,j].hist(df[df.columns[3*i+j]])
        axs[i,j].set_xlabel(df.columns[3*i+j])
'''
#plt.show()


cur= conn.cursor()
cur.execute('drop table if exists carData_table')
cur.execute('''
create table carData_table (Car_Name nvarchar(255), Year double, Selling_Price double, Present_Price double, Kms_Driven double, Fuel_Type nvarchar(255), Seller_Type nvarchar(255), Transmission nvarchar(255), Owner double)''')

    
for row in df.iterrows():
    testlist = row[1].values
    t= tuple(testlist)
    #print(t)
    cur.execute(''' INSERT INTO carData_table(Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner) VALUES(%s, %s, %s, %s, %s, %s,%s, %s, %s)''', t)

    conn.commit()
#cur.execute("select * from carData_table")

#for x in cur:
  #print(x)

df1           = pd.read_sql("select * from carData_table", conn);

 

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
#print(poly(2017))
prices_numpy= poly(df1['Year'].astype(float))
'''plt.plot(df1['Year'].astype(float), predicted_price)
plt.scatter(df1['Year'].astype(float), df1['Selling_Price'].astype(float), c='r')
plt.title('Numpy linear regression')
plt.show()
'''
#scipy linear regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df1['Year'].astype(float), df1['Selling_Price'].astype(float))
print('Scipy Linear regression parameters: ', slope, intercept, r_value, p_value, std_err)
'''plt.plot( df1['Year'].astype(float), df1['Selling_Price'].astype(float), 'o', label='original data')
plt.plot( df1['Year'].astype(float), intercept + slope* df1['Year'].astype(float), 'r', label='fitted line')
plt.legend()
plt.title('Scipy linear regression')
plt.show()
'''
#sklearn
years= df1['Year'].astype(float)
X= np.array([[year] for year in years])
print('X: ', X.shape)
Y= df1['Selling_Price'].astype(float)
reg = LinearRegression().fit(X, Y)
print(reg.score(X, Y))
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

#multiple linear regression sklearn

X_MLR = df1[['Year','Kms_Driven', 'Transmission']] 
print(df1['Transmission'].unique())
df1['Transmission'] = df1['Transmission'].replace({'Manual':'0', 'Automatic':'1'})
#df1['Transmission'] = df1['Transmission'].replace(['Automatic'],1)
print(df1['Transmission'].unique())
df1['Transmission'].astype(float)

X_MLR = df1[['Year','Kms_Driven', 'Transmission']] 

regr = LinearRegression()
regr.fit(X_MLR, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
plt.plot( df1['Year'].astype(float), df1['Selling_Price'].astype(float), 'o', label='original data')
#plt.plot( df1['Year'].astype(float), intercept + slope* df1['Year'].astype(float), 'r', label='fitted line Scipy')
#plt.plot( df1['Year'].astype(float),prices_sklearn, 'b', label='fitted line Sklearn')
#plt.plot(df1['Year'].astype(float), prices_numpy, 'g', label= 'fitted line numpy')
plt.plot(df1['Year'].astype(float), regr.predict(X_MLR), 'o r', label= 'MLR Sklearn')
print(regr.predict(X_MLR).shape)     
#plt.ylim(0, 5)
plt.legend()
plt.title('Selling price vs years')
plt.show()
rima= MyLinReg('Rima')
print(rima.SePresenter())
