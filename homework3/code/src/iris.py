from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

class MyPerceptron():
    def __init__(self,n_iter, fit_intercept = True):
        """
        LinearRegression class.
 
        Attributes
        --------------------
            fit_intercept  --  Whether the intercept should be estimated or not.
            n_iter         --  Maximum number of iterations (in case of non-converging)
            coef_          --  The learned coeffient of the linear model
        """
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
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
        y[y==0] = -1
        self.coef_ = np.zeros(d)
        for _ in range(self.n_iter):
            for index in range(n):
                if (y[index] * np.dot(self.coef_, X_[index]) <= 0):
                    self.coef_ += y[index] * X_[index]
                    break
        
        # implementation end from here ----------------------------
        # ********************************

    # def net_input(self, X):
    #     net_input = np.dot(X, self.coef_[1:]) + self.coef_[0]
    #     return net_input
    # def predict_i(self, X):
    #     #print (X,self.net_input(X))
    #     return np.where(self.net_input(X) >= 0.0, 1, -1)

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
        y = np.dot(X_, self.coef_) >= 0
        return y

iris = datasets.load_iris()
X = iris.data[:100]
y = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

# A build-in perceptron model from sikit-learn
ppn = Perceptron(max_iter=40)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
print('Training Error: %.2f' % (1-accuracy_score(y_test, y_pred)))

# MyPerceptron
myppn = MyPerceptron(n_iter = 11)
myppn.fit(X_train, y_train)
y_pred = myppn.predict(X_test)
print('Training Error: %.2f' % (1-accuracy_score(y_test, y_pred)))