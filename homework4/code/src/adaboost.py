import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from visualize import visualize_decision_boundary
import matplotlib.pyplot as plt

def sign(y):
    """
    y       -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign

class DecisionStump():
    def __init__(self,fit_intercept = True):
        """
        Decision Stump class.
 
        Attributes
        --------------------
            j          --  the learned index of decision stump
            theta      --  the learned threshold of decision stump
        """
        self.j = None
        self.theta = None
    
    def fit(self,X,D,y):
        """
        Finds the coefficients of a linear model that fits the target.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
            D       -- numpy array of shape (m,), distribution vector
            y       -- numpy array of shape (m,), targets
 
        Returns
        --------------------
            self    -- an instance of self
        """
        m,d = X.shape
        # initialization of F star
        Fs = np.inf

        # Implementation starts from here
        for j in range(d):
            # concat X[:, j], D, and y for sorting
            record = np.hstack((X[:, [j]], D.reshape(-1, 1), y.reshape(-1, 1)))
            record = np.array(sorted(record, key=lambda arr: arr[0]))

            # temp array, could visit 0, 1, ..., m
            tempArr = np.zeros(m+1)
            # retrieve sorted X[: j], D, and y
            tempArr[:m], tempD, tempY = record[:, 0], record[:, 1], record[:, 2]
            tempArr[m] = tempArr[m-1] + 1


            F = np.sum(D[y == 1])
            if  F < Fs:
                Fs = F
                self.theta = tempArr[0] - 1
                self.j = j
            
            for i in range(m):
                F -= tempY[i] * tempD[i]
                if (F < Fs) and (tempArr[i] != tempArr[i+1]):
                    Fs = F
                    self.theta = 0.5 * (tempArr[i] + tempArr[i+1])
                    self.j = j

                
        # Implementation ends from here
        
    def predict(self,X):
        """
        Predict output for X.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (m,), predictions
        """
        if (self.j is None) or (self.theta is None):
            raise Exception("Fit function not implemented")

        y = sign(self.theta - X[:,self.j])
        return y

class MyAbaboost():
    def __init__(self,n_estimators, base_estimator = DecisionStump):
        """
        Decision Stump class.
 
        Attributes
        --------------------
            n_estimators        --  The maximum number of estimators at which boosting is terminated.
            base_estimator      --  The base estimator from which the boosted ensemble is built. By default
                                    we will use DecisionStump developed above
            weak_learners       --  A list of weak learner objects (of type base_estimator)
            w                   --  A list of weights corresponding to weak learners.
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.weak_learners = []
        self.w = []
    
    def fit(self,X_train,y_train,X_test,y_test,verbose = False):
        """
        Finds the coefficients of a linear model that fits the target.
 
        Parameters
        --------------------
            X_train       -- numpy array of shape (m1,d), training features
            y_train       -- numpy array of shape (m1,), training targets
            X_test        -- numpy array of shape (m2,d), testing features (for calculating testing error per iteration only)
            y_test        -- numpy array of shape (m2,), testing targets (for calculating testing error per iteration only)
            verbose       -- boolean, for debugging purposes
 
        Returns
        --------------------
            self    -- an instance of self
        """
        m,d = X_train.shape
        D = np.array([1./m] * m)
        err_train = []        # training error per iteration
        err_test = []        # testing error per iteration

        for i in range(self.n_estimators):
            ht = self.base_estimator()
            ht.fit(X_train, D, y_train)
            
            # Implementation starts from here
            y_pred = ht.predict(X_train)
            epsilon = np.sum(D[y_pred != y_train])
            w = 0.5 * np.log(1 / (epsilon + np.finfo(np.float32).eps) - 1)
            tempArr = D * np.exp(-w * y_pred * y_train)
            for j in range(m):
                D[j] = tempArr[j] / np.sum(tempArr)
            # Implementation ends from here

            self.w.append(w)
            self.weak_learners.append(ht)

            y_pred = self.predict(X_train)
            err_train.append(1-accuracy_score(y_train, y_pred))
            y_pred = self.predict(X_test)
            err_test.append(1-accuracy_score(y_test, y_pred))

        # debugging
        if verbose :
            plt.plot(range(self.n_estimators), err_train, 'b-',label = 'Training error')
            plt.plot(range(self.n_estimators), err_test, 'r-',label = 'Testing error')
            plt.title('Adaboost: Training and Testing error w.r.t # iterations')
            plt.legend()
            plt.show()
        
    def predict(self,X):
        """
        Predict output for X.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (m,), predictions
        """
        if (self.weak_learners == []) or (self.w == []):
            raise Exception("Fit function not implemented")

        m,d = X.shape
        y = np.zeros(m,)
        for i in range(len(self.w)):
            y += self.weak_learners[i].predict(X) * self.w[i]
        return sign(y)

# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and fit an AdaBoosted decision stump
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(X_train, y_train)
y_pred = bdt.predict(X_train)
print('Training error of sklearn AdaboostClassifier: %.2f' % (1-accuracy_score(y_train, y_pred)))
y_pred = bdt.predict(X_test)
print('Testing error of sklearn AdaboostClassifier: %.2f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purposes)
# visualize_decision_boundary(ds,X_train,y_train)

# Map all labels 0 to -1
y_train[y_train==0] = -1
y_test[y_test==0] = -1

# Decision Stump
m,d = X_train.shape
D_train = np.array([1./m] * m)  # An uniform distribution throughout the training set
ds = DecisionStump()
ds.fit(X_train,D_train,y_train)
y_pred = ds.predict(X_train)
print('Training error of a single decision stump: %.2f' % (1-accuracy_score(y_train, y_pred)))
y_pred = ds.predict(X_test)
print('Testing error of a single decision stump: %.2f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purposes)
# visualize_decision_boundary(ds,X_train,y_train)

# Adaboost
ad = MyAbaboost(n_estimators = 200)
ad.fit(X_train,y_train,X_test,y_test,verbose = True)
y_pred = ad.predict(X_train)
print('Training error of Adaboost: %.2f' % (1-accuracy_score(y_train, y_pred)))
y_pred = ad.predict(X_test)
print('Testing error of Adaboost: %.2f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purposes)
# visualize_decision_boundary(ds,X_train,y_train)