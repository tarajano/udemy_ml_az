import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def scaler(X, y):
	from sklearn.preprocessing import StandardScaler
	sc_X = StandardScaler()
	X = sc_X.fit_transform(X)
	sc_y = StandardScaler()
	y = sc_y.fit_transform(y.reshape(-1, 1))
	return [X, y]

def lm_reg(X, y, scale=False):
	## Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	''' Optional Feature Scaling '''
	if scale:
		X_train, y_train = scaler(X_train, y_train)

	## Fitting Linear Regression to the Training set
	from sklearn.linear_model import LinearRegression
	classifier = LinearRegression()
	classifier.fit(X_train, y_train)
	print('R^2:', classifier.score(X_train, y_train))

	# y_pred = classifier.predict(X_test)
	# plt.plot(y_test, y_pred)
	# plt.show()

def poly_reg(X, y, scale=False):
	## Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	''' Optional Feature Scaling '''
	if scale:
		X_train, y_train = scaler(X_train, y_train)

	## Fits Linear Regression to the Training set
	from sklearn.linear_model import LinearRegression
	slm_classifier = LinearRegression()
	slm_classifier.fit(X_train, y_train)
 	
	## Transforms features into polynomial
	from sklearn.preprocessing import PolynomialFeatures
	poly_reg = PolynomialFeatures(degree=4)
	# Adds additional polynomial terms (transaformed features) to the features matrix
	X_poly = poly_reg.fit_transform(X_train)

	## Fits the polynomial features to a LinearRegression 
	#  Needs to use a different LinearRegression obj
	slm_classifier2 = LinearRegression()
	slm_classifier2.fit(X_poly, y_train)
	print('R^2:', slm_classifier2.score(X_poly, y_train))

def vif_calculation(dtf):
	"""
	Description
	-----------
		Computing variance inflation factor (VIF)

		In statistics, the variance inflation factor (VIF) is the ratio of variance in a model
		with multiple terms, divided by the variance of a model with one term alone. It quantifies
		the severity of multicollinearity in an ordinary least squares regression analysis.
	
	Implementation:
	    https://etav.github.io/python/vif_factor_python.html (code below)

	Interpretation: 
	    https://www.displayr.com/variance-inflation-factors-vifs/
	    A value of 1 means that the predictor is not correlated with other variables.
	    The higher the value, the greater the correlation of the variable with other variables.
	    Values of more than 4 or 5 are sometimes regarded as being moderate to high, with values
	    of 10 or more being regarded as very high. These numbers are just rules of thumb ...
	"""
	from statsmodels.stats.outliers_influence import variance_inflation_factor

	vif_res_dtf = pd.DataFrame()
	vif_res_dtf["features"] = dtf.columns
	vif_res_dtf['vif_factors'] = [variance_inflation_factor(dtf.values, i) for i in range(dtf.shape[1])]
	
	return vif_res_dtf

def viz(dtf):
	## Splits dtf by category
	discp_dct = {k: v for k, v in dtf.groupby('discipline')}
	gendr_dct = {k: v for k, v in dtf.groupby('sex')}

	plt.subplot(1, 2, 1)
	plt.scatter(discp_dct['A']['yrs_service'], discp_dct['A']['salary'], label='A', c='r')
	plt.scatter(discp_dct['B']['yrs_service'], discp_dct['B']['salary'], label='B', c='b')
	plt.title('Discipline')
	plt.xlabel('Years in Service')
	plt.ylabel('Salary')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.scatter(gendr_dct['Male']['yrs_service'], gendr_dct['Male']['salary'], label='Male')
	plt.scatter(gendr_dct['Female']['yrs_service'], gendr_dct['Female']['salary'], label='Female')
	plt.title('Gender')
	plt.xlabel('Years in Service')
	# plt.ylabel('Salary')
	plt.legend()
	plt.show()

def stats_mod(data):
	import statsmodels.formula.api as smf

	lm1 = smf.ols(formula='salary ~ yrs_since_phd + yrs_service + rank_enc + sex_Female + discipline_B',
	              data=data)
	results = lm1.fit()
	return results


''' Loads data '''
ini_dtf = pd.read_csv('data/SalariesSalariesForProfessors.csv')
ini_dtf.rename(columns={'yrs.since.phd': 'yrs_since_phd','yrs.service': 'yrs_service'},
               inplace=True)
ini_dtf.drop(ini_dtf.columns[0], axis=1, inplace=True)

''' Collects and drops salaries (y) '''
salaries_dtf = pd.DataFrame(ini_dtf['salary'], columns=['salary'])
salaries_y = ini_dtf.iloc[:,-1].values
ini_dtf.drop(columns=['salary'], axis=1, inplace=True)

''' Encodes categories '''
rank_dct = {'AsstProf': 1, 'AssocProf': 2, 'Prof': 3}
ini_dtf['rank_enc'] = ini_dtf['rank'].map(rank_dct)
dummies_dtf = pd.get_dummies(ini_dtf[['sex','discipline']])

''' Drops columns of encoded categories '''
ini_dtf.drop(columns=['rank', 'sex', 'discipline'], axis=1, inplace=True)

''' Drops columns to avoid dummy trap '''
dummies_dtf.drop(columns=['sex_Male','discipline_A'], axis=1, inplace=True)

''' Creates the Xs dataframe '''
X_dtf = pd.concat([ini_dtf, dummies_dtf], axis=1)

''' Estimates collinearity '''
# sys.path.append('/Users/tarajano/DevLibs/python')
# import regression_func_utils as rfu
# vif_dtf = rfu.compute_vif(X_dtf)
#         features  vif_factors
# 0  yrs_since_phd    30.021873
# 1    yrs_service    16.662139
# 2       rank_enc    11.248470
# 3     sex_Female     1.095320
# 4   discipline_B     2.035328

''' Scaling has negligible influence as independent variables have similar magnitudes '''
print('lm_reg')
lm_reg(X_dtf, salaries_y)	# R^2: 0.42 (both scaled and not scaled)
print('poly_reg')
poly_reg(X_dtf, salaries_y)	# R^2: 0.62 (both scaled and not scaled)
print('stats_mod')
results = stats_mod(pd.concat([X_dtf, salaries_dtf], axis=1))
print(results.rsquared)
# print(results.summary())
