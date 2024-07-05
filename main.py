import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv("melb_data.csv")
inputFeatures=['Landsize', 'Bathroom', 'Rooms']
X,y = df[inputFeatures], df['Price']

imputer= SimpleImputer(missing_values =pd.NA, strategy='mean')
X= imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
st_x= StandardScaler() 
X_train= st_x.fit_transform(X_train) 
X_test= st_x.transform(X_test)  
model = LinearRegression()
model.fit(X_train, y_train)
predict_price = model.predict(X_test)
print(predict_price)
print('Train Score: ', model.score(X_train, y_train)*100,'%')  
print('Test Score: ', model.score(X_test, y_test)*100,'%')  