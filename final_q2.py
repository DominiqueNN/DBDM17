import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from astropy.io.votable import parse_single_table
from sklearn.neighbors import KNeighborsRegressor

def find_error(y_true,y_pred):
	error = np.median(np.abs((y_true-y_pred)/(1+y_true)))
	return error

"""""""""
No test set
"""""""""
table = parse_single_table("Tables/PhotoZFileA.vot")
data = table.array

X = np.empty([0,len(data)])
features = ['mag_r','u-g','g-r','r-i','i-z']
for feat in features:
	X = np.append(X,[data[feat]],axis=0)
X = X.T
y_true = data['z_spec']
errors=np.array([])

alphas = np.arange(0,1,.0001)
for alpha in alphas:
	model = Lasso(alpha = alpha)
	model.fit(X,y_true)
	y_pred = model.predict(X)
	errors = np.append(errors,find_error(y_true,y_pred))

alpha = alphas[np.argmin(errors)]
model = Lasso(alpha = alpha)
model.fit(X,y_true)
y_pred = model.predict(X)

print alpha, find_error(y_true,y_pred)


"""""""""
Test set
"""""""""
tableB = parse_single_table("Tables/PhotoZFileB.vot")
dataB = tableB.array
XB = np.empty([0,len(dataB)])
for feat in features:
	XB = np.append(XB,[dataB[feat]],axis=0)
XB = XB.T
yB_true = dataB['z_spec']

yC_true = yB_true[len(dataB)/2:] #test
yB_true = yB_true[:len(dataB)/2] #validation
XC = XB[len(dataB)/2:,:]
XB = XB[:len(dataB)/2,:]

errors = np.array([])
for alpha in alphas:
	model = Lasso(alpha=alpha)
	model.fit(X,y_true)
	yB_pred = model.predict(XB)
	errors = np.append(errors,find_error(yB_true,yB_pred))

alpha = alphas[np.argmin(errors)]
model = Lasso(alpha=alpha)
model.fit(X,y_true)
yC_pred = model.predict(XC)
print alpha, find_error(yC_true,yC_pred)


"""""""""
K Nearest Neighbours
"""""""""
errors = np.array([])
for n_neighbours in np.arange(1,20):
	knn = KNeighborsRegressor(n_neighbours, weights='distance')
	knn.fit(X,y_true)
	yB_pred = knn.predict(XB)
	errors = np.append(errors,find_error(yB_true,yB_pred))
nn = np.argmin(errors)+1
knn = KNeighborsRegressor(nn, weights='distance')
knn.fit(X,y_true)
yC_pred = knn.predict(XC)
print nn, find_error(yC_true,yC_pred)
