def doCrossValidation (D, k, h):

    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    h_best=INT_MAX
    acc=0
    for fold in range(k):
        # Get all indexes for this fold
        testIdxs = idxs[fold,:]
        # Get all the other indexes
        trainIdxs = np.array(set(allIdxs) - set(testIdxs)).flatten()
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h)
        accuracy=testModel(model, D[testIdxs])
        if accuracy>acc:
            acc=accuracy
            h_best=h
 
    return h_best

# H is the list of list of hyperparameters


def doDoubleCrossValidation(D, k, H):
    
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)

    accuracies = []
    
    #Outer loop 
    for outer_fold in range(k):
        # Get all indexes for this fold
        testIdxs_outer = idxs[outer_fold,:]
        # Get all the other indexes
        trainIdxs_outer = idxs[list(set(allIdxs) - set(testIdxs)),:].flatten()
        # Train the model on the training data


        h_best = doCrossValidation(D[trainIdxs_outer], k, H[outer_fold])
        accuracies.append(testModel(model, D[testIdxs_outer]))

    return np.mean(accuracies)


        