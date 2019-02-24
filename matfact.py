# Solutions for problem 5.1 on project 2 of CS 155
# Authors: Eric Han and Eva Scheller

import numpy as np
import matplotlib.pyplot as plt

def make_scatter(x, y, xLabel, yLabel, genTitle):
    plt.scatter(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(genTitle)
    plt.show()

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    reg_term = reg * Ui
    y_diff = Yij - np.dot(Ui, Vj) 
    grad = reg_term + y_diff * (-1 * Vj)
    grad *= eta

    return grad

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    reg_term = reg * Vj
    y_diff = Yij - np.dot(Ui, Vj) 
    grad = reg_term + y_diff * (-1 * Ui)
    grad *= eta

    return grad

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the root mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """

    reg_term = (reg / 2) * (np.linalg.norm(U, ord='fro')**2 + np.linalg.norm(V, ord='fro')**2)

    sq_loss_term = 0
    for i in range(len(Y)):
        Y_pt = Y[i][2]
        U_row = U[Y[i][0]-1]
        V_col = V[Y[i][1]-1]

        sq_loss_term += (Y_pt - np.dot(U_row, V_col))**2
    sq_loss_term *= (1/2) 

    loss = (reg_term + sq_loss_term) / len(Y)

    return loss

def initMatrix(N, D, interval):
    mat = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            mat[i][j] = np.random.uniform(interval[0], interval[1])
    return mat

# if the change in error is small enough, return true. else false.
def determine_errordiff_stop(errors, diff_threshold):
    if(
        len(errors) > 2 and 
        abs((errors[-2] - errors[-1])) / abs((errors[0] - errors[1])) < diff_threshold
    ):
        print("secondary Stopping condition reached")
        return True
    else:
        return False

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """

    # didn't implement stopping after improvement decreases under threshold
    U = initMatrix(M, K, [-0.5, 0.5])
    V = initMatrix(N, K, [-0.5, 0.5])

    err = []
    err.append(get_err(U, V, Y, reg=reg))
    for i in range(max_epochs):
        print("Epoch {} in max epochs {}".format(i, max_epochs))
        np.random.shuffle(Y) # shuffle the train set before each epoch
        for j in range(len(Y)):
            # finds the gradient for this U_row and V_col
            Y_pt = Y[j][2]
            U_row = U[Y[j][0]-1]
            V_col = V[Y[j][1]-1]
            gradU = grad_U(U_row, Y_pt, V_col, reg, eta)
            gradV = grad_V(V_col, Y_pt, U_row, reg, eta)

            U[Y[j][0]-1] -= gradU
            V[Y[j][1]-1] -= gradV
        this_error = get_err(U, V, Y, reg=reg)
        err.append(this_error)
        if(determine_errordiff_stop(err, eps)):
            break
    err = np.array(err)
    return (U, V, err[-1])

# main function
if __name__ == '__main__':
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20 # use this bc told so
    reg = 0.1 # in set 5, the lowest Eout consistently was from 0.1 reg
    eta = 0.03 # learning rate
    
    # Use to compute Ein and Eout
    U,V, E_in = train_model(M, N, K, eta, reg, Y_train)
    E_out = get_err(U, V, Y_test)
    
    print(E_in, E_out)

    # A_u, sigma_u, B_u = np.linalg.svd(U)
    V = np.transpose(V)
    A_v, sigma_v, B_v = np.linalg.svd(V)

    print(A_v.shape, V.shape)

    # 2D projection of U and V
    # U_proj = np.matmul(A_u[:,0:2], U)
    V_proj = np.matmul(np.transpose(A_v[:, 0:2]), V)
    print(len(V_proj[0]), len(V_proj[1]))

    make_scatter(V_proj[0], V_proj[1], 'V col 1', 'V col 2', '2D V Projection of All Movies')


    

    

