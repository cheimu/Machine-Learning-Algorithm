from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

class MyLinearRegression():
    def __init__(self,fit_intercept = True):
        """
        LinearRegression class.
 
        Attributes
        --------------------
            fit_intercept  --  Whether the intercept should be estimated or not.
            coef_          --  The learned coeffient of the linear model
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def generate_features(self,X):
        """
        Returns pre-processed input data
        """
        if self.fit_intercept:
            ones = np.ones((len(X),1))
            return np.concatenate((X,ones),axis=1)
        return X
    
    def fit(self,X,y):
        """
        Finds the coefficients of a linear model that fits the target.
 
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
 
        Returns
        --------------------
            self    -- an instance of self
        """
        X_ = self.generate_features(X)
        n,d = X_.shape
        
        # ********************************
        # implementation start from here --------------------------
        #  ********************************
        #self.coef_ = np.random.randn(d)

        y[y==0] = -1
        y = y[:,np.newaxis]
        self.coef_ = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y)
        # implementation end from here ----------------------------
        # ********************************
        
    def predict(self,X):
        """
        Predict output for X.
 
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception("fit function not implemented")

        X_ = self.generate_features(X)
        y = np.dot(X_, self.coef_)
        return y

boston = datasets.load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

# A build-in linear regression model from sikit-learn
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Mean Squared Error: %.2f' % mean_squared_error(y_test, y_pred))

# MyLinearRegression
myreg = MyLinearRegression()
myreg.fit(X_train, y_train)
y_pred = myreg.predict(X_test)
print('Mean Squared Error: %.2f' % mean_squared_error(y_test,y_pred))