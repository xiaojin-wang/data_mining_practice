import numpy as np




class Node():
    """
    Implements an individual node in the Decision Tree.
    Nothing for you to implement in this class :)
    """      
    
    def __init__(self, y):
        self.y = y                 # Labels of sample assigned to that node
        self.score = np.inf        # Gini score of the node (measure of impurity)
        self.feature_col = None    # The feature used for the split (column number)
        self.threshold = None      # Threshold used of the splot (scalar value)
        self.left_child = None     # Left child of the node (of type Node)
        self.right_child = None    # Right child of the node (of type Node)
        
    def is_leaf(self):
        if self.feature_col is None:
            return True
        else:
            return False
    
        
    def __str__(self):
        if self.is_leaf() == True:
            return 'Leaf, gini: {:.3f}, #samples: {}'.format(self.score, len(self.y))
        else:
            return 'X[{}] <= {:.3f}, gini: {:.3f}, #samples: {}'.format(self.feature_col, self.threshold, self.score, len(self.y))




        



class MyDecisionTree:
    """
    Implements the Decision Tree. 
     * only binary splits
     * only numerical features
    """       
    
    
    def __init__(self, n_neighbors=1, max_depth=None, min_samples_split=2):
        
        ## Just a check if the parameter values are meaningful
        if max_depth is not None and max_depth < 1:
            raise Exception('If specified, max_depth must be greater or equal to 0')
        if min_samples_split is not None and min_samples_split < 1:
            raise Exception('If specified, min_samples_split must be greater or equal to 0')
            
        self.tree = None
        self.n_neighbors = n_neighbors
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    
    
    def calc_gini_score_node(self, y):
        """
        Calculate the Gini score (impurity) of a list of labels

        Inputs:
        - y: A numpy array of shape (N,) containing the sample labels, 
             where N is the number of samples in this node
             
        Returns:
        - Scalar value representing the Gini score of a list of labels. 
        """        
        
        gini = None

        #########################################################################################
        ### Your code starts here ###############################################################
        classes, class_counts = np.unique(y, return_counts = True)
        gini = 1 - np.sum([np.square(class_counts[i]/np.sum(class_counts)) for i in range(len(classes))])


        ### Your code ends here #################################################################
        #########################################################################################

        ## Return the final Gini score
        return gini
    
    
    
    
    def calc_gini_score_split(self, y_left, y_right):
        """
        Calculate the Gini score (impurity) of a binary split

        Inputs:
        - y_left:  A numpy array of shape (N,) containing the sample labels, 
                   where N is the number of samples in left node
        - y_right: A numpy array of shape (N,) containing the sample labels, 
                   where N is the number of samples in right node
                   
        Returns:
        - Scalar value representing the average Gini score of two lists of labels
          weighted be the size of the two lists
        """  
        
        gini_score = None

        #########################################################################################
        ### Your code starts here ###############################################################
        n = len(y_left) + len(y_right)
        gini_score = (len(y_left)/n) * self.calc_gini_score_node(y_left) + (len(y_right)/n) * self.calc_gini_score_node(y_right)
        

        ### Your code ends here #################################################################
        #########################################################################################

        return gini_score    
    
    
    
    def calc_thresholds(self, x):
        """
        Calculates the set of all valid thresholds given a list of numerical values.
        The set of all valid thresholds is a set of minimum size that contains
        the values that would split the input list of values into two sublist:
        (1) all values less or equal the the threshold
        (2) all values larger than the threshold

        Inputs:
        - x: A numpy array of shape (N,) containing N numerical values, 
             Example: x = [4, 1, 2, 1, 1, 3]
             
        Returns:
        - Set of numerical values representing the thresholds 
          Example for input above: set([1.5, 2.5, 3.5])
        """              
        
        thresholds = set()

        #########################################################################################
        ### Your code starts here ###############################################################
        values = sorted(np.unique(x))
        if len(values) > 1:
            for index in range(len(np.unique(x))-1):
                threshold = values[index] + (values[index + 1] - values[index])/2
                thresholds.add(threshold)

        ### Your code ends here #################################################################
        #########################################################################################

        return thresholds    
    
    
    
    
    def create_split(self, x, threshold):
        """
        Splits a list of numerical input values w.r.t. to a threshold into 2 new lists:
        (1) A list containing the indices of all values less or equal to the threshold
        (2) A list containing the indices of all values larger than the threshold

        Inputs:
        - x: A numpy array of shape (N,) containing N numerical values, 
             Example: x = [4, 1, 2, 1, 1, 3]
        - threshold: A numerical value
                     Example: 2.5
                   
        Returns:
        - Tuple (indices_left, indices_right) where indices_left is the list containing
          he indices of all values less or equal to the threshold, and indices_right is 
          the list containing the indices of all values larger than the threshold
          Example for input above: ([1, 2, 3, 5], [0, 5])
        """          
        
        indices_left, indices_right = None, None

        #########################################################################################
        ### Your code starts here ###############################################################
        indices_left = np.where(x <= threshold)[0].tolist()
        indices_right = np.where(x > threshold)[0].tolist()
        
        
        ### Your code ends here #################################################################
        #########################################################################################
        
        return indices_left, indices_right    
    
    
    
    def calc_best_split_feature(self, x, y):
        """
        Calculates the best split for a feature

        Inputs:
        - x: A numpy array of shape (N,) containing N numerical values
             representing the feature values
        - y: A numpy array of shape (N,) containing N labels
             
        Returns:
        - best_score: Numerical value representing the (Gini) score of the best split
        - best_threshold: Numerical value representing the used threshold for the best split
        - best_split: Tuple (indices_left, indices_right) representing the best split
        """

        ## Calculate all valid thresholds for feature x
        thresholds = self.calc_thresholds(x)

        ## Initialize the return values
        best_score, best_threshold, best_split = np.inf, None, None

        ## Check for each threshold, which split has the best (lowest) score
        for t in thresholds:

            #########################################################################################
            ### Your code starts here ###############################################################
            ind_left, ind_right = self.create_split(x, t)
            #original_score = self.calc_gini_score_node(y)
            score = self.calc_gini_score_split(y[ind_left], y[ind_right])
            if score <= best_score:
                best_score = score
                best_threshold = t
                best_split = (ind_left, ind_right)

            ### Your code ends here #################################################################
            #########################################################################################
            
            pass # Only there to avoid errors in case the loop is empty

        # Return the best split together with the relevant information
        return best_score, best_threshold, best_split
    
    
    
    def calc_best_split(self, X, y):
        """
        Calculates the best split across all features

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
        - y: A numpy array of shape (N,) containing N labels
             
        Returns:
        - best_score: Numerical value representing the (Gini) score of the best split
        - best_threshold: Numerical value representing the used threshold for the best split
        - best_split: Tuple (indices_left, indices_right) representing the best split
        """        

        ## Initialize the return values
        best_score, best_threshold, best_col, best_split = np.inf, None, None, None

        ## Check for each feature (i.e., each column in X), which split has the best (lowest) score
        for col in range(X.shape[1]):

            #########################################################################################
            ### Your code starts here ###############################################################    
            score, threshold, split = self.calc_best_split_feature(X[:,col], y)
            #if score == 0: # find the perfect split, stop iterating
                #best_score = score
                #best_threshold = threshold
                #best_split = split
                #best_col = col
                #return best_score, best_threshold, best_col, best_split
            if score <= best_score:
                best_score = score
                best_threshold = threshold
                best_split = split
                best_col = col

            ### Your code ends here #################################################################
            #########################################################################################
            
            #pass # Only there to avoid errors in case the loop is empty            

        ## Return the best split together with the relevant information
        return best_score, best_threshold, best_col, best_split
    
    
    
    def fit(self, X, y):
        """
        Trains the Decision Tree Classifier

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
        - y: A numpy array of shape (N,) containing N labels
        
        Returns:
        - nothing
        """           
        ## Initializa Decision Tree as a single root node
        self.tree = Node(y)

        ## Start recursive building of Decision Tree
        self._fit(X, y, self.tree)
        
        ## Return Decision Tree object
        return self



    def _fit(self, X, y, node, depth=0):
        """
        Trains the Decision Tree Classifier (facilitates recursion)

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
        - y: A numpy array of shape (N,) containing N labels
        - max_depth: The maximum depth of the tree. If None, then nodes are expanded
                     until all leaves are pure or until all leaves contain less than
                     min_samples_split samples (int)
        - min_samples_split: The minimum number of samples required to split an internal node
                             (int)
        - depth: Depth of tree (int)
             
        Returns:
        - nothing
        """           

        ## Calculate and set Gini score of the node itself
        node.score = self.calc_gini_score_node(y) 

        #########################################################################################
        ### Your code starts here ###############################################################
        
        if node is None or len(y) == 0: # tree stops at previous level or no data in this group
            return
       
        if self.max_depth and depth >= self.max_depth: # max depth is reached
            return
        

        ### Your code ends here #################################################################
        #########################################################################################

        ## Calculate the best split
        score, threshold, col, split = self.calc_best_split(X, y)


        #########################################################################################
        ### Your code starts here ###############################################################
        if node.score - score <= 0: # information gain required to > 0
            return
        
        if (len(split[0]) + len(split[1])) < self.min_samples_split: # node less than minimum number of samples
            return 


        ### Your code ends here #################################################################
        #########################################################################################
        
        ## Split the input and labels using the indices from the split
        X_left, X_right = X[split[0]], X[split[1]]
        y_left, y_right = y[split[0]], y[split[1]]

        ## Update the parent node based on the best split
        node.feature_col = col
        node.threshold = threshold
        node.left_child = Node(y_left)
        node.right_child = Node(y_right)

        ## Recursively fit both child nodes (left and right)
        self._fit(X_left, y_left, node.left_child, depth=depth+1)
        self._fit(X_right, y_right, node.right_child, depth=depth+1)   

        
        
        
        
        
    def predict(self, X):
        """
        Predict labels for a set of samples

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
             
        Returns:
        - y_pred: A numpy array (N,) containing the N predicted class labels (numerical labels!)
        """             
        
        ## Return list of individually predicted labels
        return np.array([ self.predict_sample(self.tree, x) for x in X ])


    def predict_sample(self, node, x):
        """
        Predict label for a single data sample

        Inputs:
        - n: A Node object
        - x: A numpy array (F,) representing a single data sample with F features
             
        Returns:
        - y: Predicted class labels (numerical label 0, 1, 2, ...)
        """           
        
        ## If the node is a leaf, return the class with highest probability
        ## (this can happen of in the leaf are still different classes)
        if node.is_leaf():
            
            #########################################################################################
            ### Your code starts here ###############################################################
            
            
            feature, counts = np.unique(node.y, return_counts=True)
            return feature[np.argmax(counts/len(node.y))]
            
            #x[node.feature_col]
        
            ### Your code ends here #################################################################
            #########################################################################################        

            pass
            
        ## If the node is not a leaf, go down the left or right subtree
        ## depending on whether the feature value <= threshold or not
        if x[node.feature_col] <= node.threshold:
            return self.predict_sample(node.left_child, x)
        else:
            return self.predict_sample(node.right_child, x)        

        
        
        
        
        
        
        
    def __str__(self):
        self.print_tree(self.tree)
        return ''
        

    def print_tree(self, node, level=0):
        print('---'*level, node)
        if node.left_child is not None:
            self.print_tree(node.left_child, level=level+1)
        if node.right_child is not None:
            self.print_tree(node.right_child, level=level+1)
        
        