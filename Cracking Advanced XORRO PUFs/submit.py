from sklearn.svm import LinearSVC
from sklearn import linear_model
import numpy as np
import time

def my_fit(Z_train):
    # extract the feature vectors, p, q, and responses
    global models
    X_train = Z_train[:, :64]
    p = Z_train[:, 67:63:-1].dot(2 ** np.arange(4))
    q = Z_train[:, 71:67:-1].dot(2 ** np.arange(4))

    y_train = Z_train[:, -1]

    models = {}
    for i in range(len(p)):
        # get the p, q pair for the current row
        p_i = p[i]
        q_i = q[i]

        # check if a model for the p, q pair already exists
        if (p_i, q_i) not in models:
            X_set = []
            y_set = []
            for j in range(len(Z_train)):
                if p[j] == p_i and q[j] == q_i:
                    X_set.append(X_train[j])
                    y_set.append(y_train[j])
                elif p[j] == q_i and q[j] == p_i:
                    X_set.append(X_train[j])
                    y_set.append(1-y_train[j])
            X_set = np.array(X_set)
            y_set = np.array(y_set)

            # train a LinearSVC model on the p, q pair
            model = LinearSVC()
            model.fit(X_set, y_set)
            models[(p_i, q_i)] = model
    return models

# define a function to make predictions on a given row of test data
def my_predict(X_test,models):
    # extract the set number from the row
    # print(X_test.shape)
    y_pred=np.zeros((X_test.shape[0],))
    
    for i in range(X_test.shape[0]):
        # extract the set number from the row
        p = X_test[i, 67:63:-1].dot(2 ** np.arange(4))
        q = X_test[i, 71:67:-1].dot(2 ** np.arange(4))
        p_q = (p, q)
        # use the corresponding model to make the prediction
        model = models[p_q]
        y_pred_i = model.predict(X_test[i, :64].reshape(1,-1))
        y_pred[i] = int(y_pred_i[0])
    
    
    return y_pred
    
# # load the training data
# data_trn = np.loadtxt("train.dat")
# # fit the models on the training data
# start = time.time()
# models = my_fit(data_trn)
# end = time.time()
# print("time taken in training = ", end-start)
# # # load the test data and true labels
# data_tst = np.loadtxt("test.dat")
# X_test = data_tst[:, :-1]
# y_true = data_tst[:, -1]

# # make predictions on the test data
# y_pred = my_predict(data_tst[:,:-1],models)
# print(y_pred.shape)
# sum = 0
# for i in range(len(y_true)):
#     if(y_true[i] == y_pred[i]):
#         sum = sum+1
# print("Accuracy = ", 100*sum/len(y_true))
