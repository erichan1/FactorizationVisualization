# Solutions for problem 5.2 on project 2 of CS 155
# Authors: Eric Han and Eva Scheller

import numpy as np
import matplotlib.pyplot as plt


'''
movie_IDs = np.array([0, 1,2,3,...]). Just put in the 1 indexed IDs. I subtract 1 later. 
V is a (N movies, K) np array. Outputted by whatever matrix factorization method you use. 

'''
def project_movies_2D(V, movie_IDs):
    V_T = np.transpose(V)
    A_v, sigma_v, B_v = np.linalg.svd(V_T)

    # 2D projection of U and V
    # U_proj = np.matmul(A_u[:,0:2], U)
    V_proj = np.matmul(np.transpose(A_v[:, 0:2]), V_T)

    # take only the columns corresponding to movie ids. 
    # movie ids are 1 indexed, so subtract by 1.
    zero_index_IDs = movie_IDs - 1
    V_proj_specific = V_proj[:, zero_index_IDs]

    return (V_proj_specific[0], V_proj_specific[1])

# X, Y are np arrays that have x and y positions of the movies
# movie_titles = np.array(['the godfather', 'star wars',...]). holds titles of movie_IDs.
# gentitle is a string that you put on the entire thing
def make_movie_scatter(X, Y, movie_titles, gentitle, has_legend):
    plt.figure(1)
    for i in range(len(X)):
        plt.plot(X[i],Y[i],'o',label=movie_titles[i])

    if(has_legend):
        plt.legend(loc='')

    plt.xlabel('V projection col 1')
    plt.ylabel('V projection col 2')
    plt.title(gentitle)
    plt.show()

# make a combo scatter 
# V is just the V
# ID_List = [[romance_id], [anime_id], [etc_id]]
# ID_titles = [genre1, genre2, genre3]
# gentitle = 'title of the graph'
def make_combo_scatter(V, ID_List, ID_titles, genTitle):

    plt.figure(1)

    for i in range(len(ID_List)):
        movie_IDs = ID_List[i] # the list of IDs to project
        title = ID_titles[i]

        X, Y = project_movies_2D(V, movie_IDs)
        plt.plot(X, Y, 'o',label=title)

    plt.legend(loc='best')
    plt.xlabel('V Projection Component 1')
    plt.ylabel('V Projection Component 2')
    plt.title(genTitle)
    # plt.xlim(min(V1_all),max(V1_all))
    # plt.ylim(min(V2_all),max(V2_all))

    plt.show()

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    reg_term = 2 * reg * Ui
    y_diff = Yij - np.dot(Ui, Vj) 
    grad = reg_term + y_diff * (-2 * Vj)
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
    reg_term = 2 * reg * Vj
    y_diff = Yij - np.dot(Ui, Vj) 
    grad = reg_term + y_diff * (-2 * Ui)
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

    reg_term = reg * (np.linalg.norm(U, ord='fro')**2 + np.linalg.norm(V, ord='fro')**2)

    sq_loss_term = 0
    for i in range(len(Y)):
        Y_pt = Y[i][2]
        U_row = U[Y[i][0]-1]
        V_col = V[Y[i][1]-1]
        sq_loss_term += (Y_pt - np.dot(U_row, V_col))**2

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
        (errors[-2] - errors[-1]) / (errors[0] - errors[1]) < diff_threshold
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

def makeSixGraphs(V):

    random_IDs = np.array([127,187,64,172,181,50,59,60,61,89])
    random_titles = ['Godfather Part I','Godfather: Part II','Shawshank Redemption',
        'Empire Strikes Back','Return of the Jedi','Star Wars','Three Colors: Red',
        'Three Colors:Blue','Three Colors: White','Blade Runner']

    Anime_ID = np.array([1,71,95,99,101,102,103,2,114,169])
    Anime_names = ['Toy Story','Lion King','Aladdin','Snow White',
        'Heavy Metal','Aristocats','All Dogs Go to Heaven 2',
        'Wallace and Gromit','Wrong Trouser','Grand Day Out, A']

    Scifi_ID = np.array([7,38,39,50,62,82,84,89,96,101])
    Scifi_names = ['Twelve Monkeys','The Net','Strange Days','Star Wars',
        'Stargate','Jurassic Park','Robert A Heinleins The Puppet Masters',
        'Blade Runner','Terminator 2','Heavy Metal']

    Romance_ID = np.array([14,16,20,33,36,49,50,51,55,66])
    Romance_names = ['Postino, II','French Twist','Angels and Insects',
        'Desperado','Mad Love','I.Q.','Star Wars','Legends of the Fall',
        'Professional, The','While You Were Sleeping']


    popular_ID = np.array([174, 121, 300, 1, 288, 286, 294, 181, 100, 258])
    popular_titles = ['Raiders of the Lost Ark (1981)', 'Independence Day (ID4) (1996)', 
        'Air Force One (1997)', 'Toy Story (1995)', 'Scream (1996)', '"English Patient, The (1996)"', 
        'Liar Liar (1997)', 'Return of the Jedi (1983)', 'Fargo (1996)', 'Contact (1997)']

    best_rated_ID = np.array([1449, 814, 1122, 1189, 1201, 1293, 1467, 1500, 1536, 1599])
    best_rated_titles = ['Pather Panchali (1955)', '"Great Day in Harlem, A (1994)"', 
        'They Made Me a Criminal (1939)', 'Prefontaine (1997)', 'Marlene Dietrich: Shadow and Light (1996) ', 
        'Star Kid (1997)', '"Saint of Fort Washington, The (1993)"', 'Santa with Muscles (1996)', 
        'Aiqing wansui (1994)', "Someone Else's America (1995)"]

    V_X, V_Y = project_movies_2D(V, random_IDs)
    make_movie_scatter(V_X, V_Y, random_titles, '2D V Projection of Ten Movies We Chose', True)
    make_movie_scatter(V_X, V_Y, random_titles, '2D V Projection of Ten Movies We Chose', False)

    V_X, V_Y = project_movies_2D(V, popular_ID)
    make_movie_scatter(V_X, V_Y, popular_titles, '2D V Projection of Ten Most Popular Movies', True)
    make_movie_scatter(V_X, V_Y, popular_titles, '2D V Projection of Ten Most Popular Movies', False)

    V_X, V_Y = project_movies_2D(V, best_rated_ID)
    make_movie_scatter(V_X, V_Y, best_rated_titles, '2D V Projection of Ten Best Rated Movies', True)
    make_movie_scatter(V_X, V_Y, best_rated_titles, '2D V Projection of Ten Best Rated Movies', False)

    V_X, V_Y = project_movies_2D(V, Anime_ID)
    make_movie_scatter(V_X, V_Y, Anime_names, '2D V Projection of Ten Animated Movies', True)
    make_movie_scatter(V_X, V_Y, Anime_names, '2D V Projection of Ten Animated Movies', False)

    V_X, V_Y = project_movies_2D(V, Scifi_ID)
    make_movie_scatter(V_X, V_Y, Scifi_names, '2D V Projection of Ten Scifi Movies', True)
    make_movie_scatter(V_X, V_Y, Scifi_names, '2D V Projection of Ten Scifi Movies', False)

    V_X, V_Y = project_movies_2D(V, Romance_ID)
    make_movie_scatter(V_X, V_Y, Romance_names, '2D V Projection of Ten Romance Movies', True)
    make_movie_scatter(V_X, V_Y, Romance_names, '2D V Projection of Ten Romance Movies', False)

    make_combo_scatter(V, [Romance_ID, Scifi_ID, Anime_ID], 
        ['Romance', 'Scifi', 'Anime'], 'Three Genre Movie Comparison Projection')

    make_combo_scatter(V, [popular_ID, best_rated_ID], 
        ['Most Popular', 'Best Rated'], 'Best Rated vs. Most Popular Projection')

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

    makeSixGraphs(V)

    
