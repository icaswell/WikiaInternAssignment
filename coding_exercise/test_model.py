
def accuracy(model, X_train, y_train, X_test, y_test):
    model.train_model(X_train, y_train)
    y_pred = model.predictions(X_test)
    return sum(y_test==y_pred)/len(y_test)*1.0

def held_out_accuracy(model, X, y, hold_out = .3):
    """
    X - a Nxk matrix, where each row corresponds to a training example
    y - a Nx1 vector of class labels for the instances
    """
    N = X.shape[0]
    X_test = X[0:N*hold_out, :]
    y_test = y[0:N*hold_out]
    X_train = X[N*hold_out:, :]
    y_train = y[N*hold_out:]
    return accuracy(model, X_train, y_train, X_test, y_test)


def LOOCV(model, X, y):
    """
    X - a Nxk matrix, where each row corresponds to a training example
    y - a Nx1 vector of labels for the instances
    
    returns a double corresponding to the leave-out-one cross validation score for a dataset X with labels y, 
    predicted using the model model.
    Good for small datasets.
    """
    return LOMCV(model, X, y, 1)