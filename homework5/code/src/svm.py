import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from visualize import plot_decision_boundary
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

def sign(y):
    """
    y       -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign

class RBF():
    def __init__(self, gamma):
        """
        RBF kernel class.
 
        Attributes
        --------------------
            gamma            -- scalar, hyperparameter gamma
        """
        self.gamma = gamma

    def __call__(self, X, Y):
        """
        Finds the coefficients of a linear model that fits the target.
 
        Parameters
        --------------------
            X       -- numpy array of shape (mx,d), left argument of the returned kernel k(X,Y)
            Y       -- numpy array of shape (my,d), right argument of the returned kernel k(X,Y)
        
        Returns
        --------------------
            K       -- numpy array of shape (mx,my), kernel K(X,Y)
        """
        mx,_ = X.shape
        my,_ = Y.shape
        K = np.zeros((mx,my))

        # Implementation starts from here
        for i in range(mx):
            for j in range(my):
                K[i, j] += np.exp(-self.gamma * np.square(np.linalg.norm(X[i] - Y[j])))
        # Implementation ends from here
        return K

class LinearSVM():
    def __init__(self, lambda_ = 1., T = 500, fit_intercept = True):
        """
        Linear SVM class.
 
        Attributes
        --------------------
            w                --  The learned weights of SVM
            T                --  Maximum # of iteration
            lambda_          --  Hyperparameter lambda in Eq 15.12
            fit_intercept    --  Whether the intercept should be estimated or not.
        """
        self.w = None
        self.lambda_ = lambda_
        self.T = T
        self.fit_intercept = fit_intercept

    def generate_features(self,X):
        """
        Returns pre-processed input data
        """
        if self.fit_intercept:
            ones = np.ones((len(X),1))
            return np.concatenate((X,ones),axis=1)
        return X

    def objective(self, X, y, w):
        """
        Returns the value of objective function defined in Eq 15.12
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d+self.intercept), features
            y       -- numpy array of shape (m,), targets
            w       -- current weights
        
        Returns
        --------------------
            cost    -- the value of objective function
        """
        m,_ = X.shape
        hinge_loss = np.maximum(np.zeros(m), 1 - y * np.dot(X, w))
        return self.lambda_ / 2 * np.dot(w,w) + 1./m * np.sum(hinge_loss)
    
    def fit(self,X,y,verbose = False):
        """
        Finds the coefficients of a linear model that fits the target.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
            y       -- numpy array of shape (m,), targets
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        X_ = self.generate_features(X)
        m,d = X_.shape
        obj_value = []          # keep a record of the objective per iteration
        self.w = np.zeros(d)    # keep a running average over updated weights
        theta = np.zeros(d)     # initialize theta

        for t in range(1,self.T):
            # Implementation starts from here
            w = ((1 / self.lambda_) * theta)[np.newaxis,:]
            i = np.random.choice(m)
            X_i = (X_[i])[:,np.newaxis]

            if y[i] * w.dot(X_i) < 1:
                theta += (y[i] * X_i).squeeze()

            # Implementation ends from here

            self.w = float(t-1)/t * self.w + 1./t * w           # update the average
            self.w = self.w.squeeze()
            obj_value.append(self.objective(X_, y, self.w))     # compute objective function

        # debugging. 
        if verbose :
            plt.plot(range(1,self.T), obj_value, 'b-',label = 'Objective function')
            plt.title('SVM: Objective Function')
            plt.xlabel('iteration')
            plt.ylabel('objective function')
            plt.show()
        
    def predict(self, X):
        """
        Predict output for X.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (m,), predictions
        """
        if (self.w is None):
            raise Exception("Fit function not implemented")

        X_ = self.generate_features(X)
        y = sign(np.dot(X_, self.w))
        return y

    def decision_function(self, X):
        """
        Distance of the samples X to the separating hyperplane.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (m,), distance to the separating hyperplane
        """
        if (self.w is None):
            raise Exception("Fit function not implemented")

        X_ = self.generate_features(X)
        y = np.dot(X_, self.w)
        return y

class RBFSVM():
    def __init__(self, gamma = 2, T = 500, lambda_= .01):
        """
        SVM class.
 
        Attributes
        --------------------
            RBF              -- the kernel
            T                -- maximum # of interation
            lambda_          -- hyperparameter lambda in Eq 15.12
            alpha            -- numpy array of shape (m,), the learned coefficients
            sv               -- numpy array of shape (m',d), the learned support vectors
        """
        self.RBF = RBF(gamma)
        self.lambda_ = lambda_
        self.T = T
        self.alpha = None
        self.sv = None

    def dual_objective(self, K, y, alpha):
        """
        Returns the value of dual objective function defined in Eq 15.11
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
            y       -- numpy array of shape (m,), targets
            alpha   -- current alpha
        
        Returns
        --------------------
            dual_objective    -- the value of dual objective function
        """
        K_ = np.outer(alpha * y,alpha * y)
        K_ = np.multiply(K_, K)
        return np.sum(alpha) - 0.5 * np.sum(K_)
    
    def fit(self,X,y,verbose = True):
        """
        Finds the coefficients of a RBF model that fits the target.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
            y       -- numpy array of shape (m,), targets
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        m,d = X.shape
        dual_obj = []               # keep a record of dual objective per iteration
        K = self.RBF(X,X)           # generate the kernel matrix K(X,X)
        self.alpha = np.zeros(m)    # keep a running average over updated alpha
        beta = np.zeros(m)          # initialize beta
        
        for t in range(1,self.T):
            # Implementation starts from here
            alpha = (1 / (self.lambda_*t)) * beta
            i = np.random.choice(m)      
            if y[i] * np.sum(alpha.dot(K[:,i])) < 1:
                beta[i] += y[i]
            # Implementation ends from here

            self.alpha = float(t-1)/float(t) * self.alpha + 1./float(t) * alpha # update the average
            dual_obj.append(self.dual_objective(K, y, self.alpha/y)) # evaluate the dual objective
        
        index = (self.alpha != 0) 
        self.sv = X[index]                       # store all support vectors
        self.alpha = self.alpha[index]           # store all non-zero coefficients

        # debugging
        if verbose :
            plt.plot(range(1,self.T), dual_obj, 'b-',label = 'Dual Objective function')
            plt.xlabel('iteration')
            plt.ylabel('dual objective function')
            plt.title('RBF SVM: Dual Objective Function')
            plt.show()
        
    def predict(self, X):
        """
        Predict output for X.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (m,), predictions
        """
        if (self.alpha is None):
            raise Exception("Fit function not implemented")

        K = self.RBF(X, self.sv)
        y = sign(np.dot(K,self.alpha))
        return y

    def decision_function(self, X):
        """
        Distance of the samples X to the decision boundary.
 
        Parameters
        --------------------
            X       -- numpy array of shape (m,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (m,), distance to the decision boundary
        """
        if (self.alpha is None):
            raise Exception("Fit function not implemented")

        K = self.RBF(X, self.sv)
        y = np.dot(K,self.alpha)
        return y

# Construct dataset. DO NOT change.
X, y = make_moons(noise=0.3, random_state=0)
# map 0 label to -1
y[y==0] = -1
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

########### Question 7 ###########

# fit a linear model using scikit-learn SVC
clf = svm.SVC(kernel='linear', C=0.025)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training error of scikit-learn Linear SVM: %.3f' % (1-accuracy_score(y_train, y_pred)))
y_pred = clf.predict(X_test)
print('Testing error of scikit-learn Linear SVM: %.3f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purpose)
# plot_decision_boundary(clf, 1,X, y, X_test, y_test, 'Decision Boundary of Scikit-learn Linear SVM')
# plt.show()

# fit LinearSVM
clf = LinearSVM(lambda_ = 0.4, T = 5000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training error of LinearSVM: %.3f' % (1-accuracy_score(y_train, y_pred)))
y_pred = clf.predict(X_test)
print('Testing error of LinearSVM: %.3f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purpose)
#plot_decision_boundary(clf, 2, X, y, X_test, y_test, 'Decision Boundary of LinearSVM')
#plt.show()

########### Question 8 ###########

# fit a RBF model using scikit-learn SVC
clf = svm.SVC(kernel='rbf', C=1. , gamma = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training error of scikit-learn RBF SVM: %.3f' % (1-accuracy_score(y_train, y_pred)))
y_pred = clf.predict(X_test)
print('Testing error of scikit-learn RBF SVM: %.3f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purpose)
#plot_decision_boundary(clf, 3, X, y, X_test, y_test, 'Decision Boundary of Scikit-learn RBF SVM')
#plt.show()

# fit RBFSVM
clf = RBFSVM()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training error of RBFSVM: %.3f' % (1-accuracy_score(y_train, y_pred)))
y_pred = clf.predict(X_test)
print('Testing error of RBFSVM: %.3f' % (1-accuracy_score(y_test, y_pred)))
# Uncomment to visualize decision boundary (not required for the homework, for debugging purpose)
plot_decision_boundary(clf, 4,X, y, X_test, y_test, 'Decision Boundary of RBFSVM')
plt.show()
