import numpy as np

from sklearn.metrics import mean_squared_error, log_loss


class MyLinearRegression:
    
    
    def __init__(self):
        self.theta = None
    
    
    def loss(self, y_true, y_pred, squared=True):
        return mean_squared_error(y_true, y_pred, squared=squared)
    
    
    def add_bias(self, X):
        """
        Adds the constant/bias term a 1st colum to data matrix X

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
             
        Returns:
        - X_bias: A numpy array of shape (N, F+1) containing N data samples with F features
                  and the a "constant feature" of 1 in the first column
        """          

        X_bias = None

        #########################################################################################
        ### Your code starts here ###############################################################    
        
        X_bias = np.c_[np.ones((X.shape[0],1)),X]

        ### Your code ends here #################################################################
        #########################################################################################    

        return X_bias    
    
    
    def fit_analytically(self, X, y):
        """
        Computes the Normal Equation to find the best theta values analytically

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
        - y: A numpy array of shape (N,) containing N ground truth values
             
        Returns:
        - nothing (bet sets self.theta which should be a numpy array of shape (F+1,)
          containing the F+1 coefficients for the F features and the constant/bias term)
        """         

        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        theta = None

        #########################################################################################
        ### Your code starts here ###############################################################

        theta = np.dot(np.linalg.pinv(X),y) # pinv: pseudo inverse

        ### Your code ends here #################################################################
        #########################################################################################

        self.theta = theta
    
    
    
    def calc_h(self, X):
        """
        Calculates the predcitions h_theta(x) for each data sample x

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
             
        Returns:
        - h: A numpy array of shape (N,) containing N predictions
        """            

        h = None

        #########################################################################################
        ### Your code starts here ###############################################################    
        h = np.dot(X,self.theta)

        ### Your code ends here #################################################################
        #########################################################################################    

        return h


    def calc_gradient(self, X, y, h):
        """
        Calculates the gradient w.r.t to all theta_i

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
        - y: A numpy array of shape (N,) containing N ground truth values
        - h: A numpy array of shape (N,) containing N predictions
             
        Returns:
        - grad: A numpy array of shape (F+1,) -- F features and the constant/bias term
        """

        grad = None

        #########################################################################################
        ### Your code starts here ###############################################################  
        errors = np.subtract(h, y)
        grad = (2/X.shape[0]) * np.dot(X.transpose(), errors)
       
        ### Your code ends here #################################################################
        #########################################################################################

        return grad
    
    
    def fit(self, X, y, lr=0.001, num_iter=1000, verbose=True):
        """
        Fits a Linear Regression model on a given dataset

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
        - y: A numpy array of shape (N,) containing N ground truth values
        - lr: A real value representing the learning rate
        - num_iter: A integer value representing the number of iterations 
        - verbose: A Boolean value to turn on/off debug output
             
        Returns:
        - nothing (bet sets self.theta which should be a numpy array of shape (F+1,)
          containing the F+1 coefficients for the F features and the constant/bias term)
        """
        
        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1]).reshape(-1,1)

        for i in range(num_iter):

            #########################################################################################
            ### Your code starts here ###############################################################      
            h = self.calc_h(X)
            grad = self.calc_gradient(X, y, h)
            self.theta = self.theta - lr * grad
            
            ### Your code ends here #################################################################
            #########################################################################################        

            # Print loss every 10% of the iterations
            if verbose == True:
                if(i % (num_iter/10) == 0):
                    print('Loss: {:.3f} \t {:.0f}%'.format(self.loss(h, y), (i / (num_iter/100))))

        # Print final loss
        print('Loss: {:.3f} \t 100%'.format(self.loss(h, y)))
    
    
    
    def predict(self, X):
        """
        Predicts the values for a set of data samples

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
             
        Returns:
        - A numpy array of shape (N,) containing N predicted values
        """
        
        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        return self.calc_h(X)
    