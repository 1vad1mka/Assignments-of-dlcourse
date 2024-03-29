import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    
    cond = (predictions.ndim == 1)
    
    if cond:
        predictions = predictions.reshape(1, predictions.shape[0])
        
    predictions_new = predictions - np.max(predictions, axis=1).reshape(predictions.shape[0], 1)
    
    predictions_new_exp = np.exp(predictions_new)
    predictions_new_exp_sum = np.sum(predictions_new_exp, axis=1).reshape(predictions_new_exp.shape[0], 1)
    result = (predictions_new_exp / predictions_new_exp_sum)
    
    if cond:
        result = result.reshape(result.size)
    
    
    return result
   
    

    



def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    cond = (probs.ndim == 1)
    
    if cond:
        probs = probs.reshape(1, probs.shape[0])
        target_index = np.array([target_index])

    rows = np.arange(probs.shape[0])
    columns = target_index
    
    return np.mean(-np.log(probs[rows, columns]))



def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    cond = (predictions.ndim == 1)
    
    if cond:
        predictions = predictions.reshape(1, predictions.shape[0])
        target_index = np.array([target_index ])
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    onehot_index = np.zeros(predictions.shape)
    onehot_index[np.arange(predictions.shape[0]), target_index] = 1
    
    # dL/dZ
    dprediction = (probs - onehot_index) / probs.shape[0]
    
    if cond:
        dprediction = dprediction.reshape(dprediction.size)
        
    return loss, dprediction

    

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.trace(np.matmul(W.T, W))
    grad = 2 * reg_strength * W
    

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)   # L, dL/dZ

    dW = np.matmul(dprediction.T, X).T  # dL/dW = (dL/dZ).T * X

    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            
            loss = 0
            for batch_indx in batches_indices:
                X_batch = X[batch_indx, :]
                y_batch = y[batch_indx]
                
                lin_loss, lin_dW = linear_softmax(X_batch, self.W, y_batch)
                reg_loss, reg_dW = l2_regularization(self.W, reg)
            
                loss = lin_loss + reg_loss
                dW = lin_dW + reg_dW
                
                self.W -= learning_rate * dW
                
            loss_history.append(loss)          


        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        Z = np.dot(X, self.W)
        S = softmax(Z)
        y_pred = np.argmax(S, axis=1)

        return y_pred



                
                                                          

            

                
