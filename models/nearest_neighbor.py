from sklearn.metrics.pairwise import cosine_similarity

def nearest_neighbor(matrix_train, **unused):
    m, n = matrix_train.shape

    RQ = matrix_train
    Y = cosine_similarity(X=matrix_train.T, Y=None, dense_output=False)

    return RQ, Y, None
