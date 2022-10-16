import numpy as np

#function to find activation for training set
def train(X, label, weights):
    activation = np.dot(X, weights[1:]) + weights[0]    #using first element to store bias
    return 1.0 if activation*label > 0.0 else 0.0

#function to train perceptron
def perceptron_train(X,Y):
    weights = np.zeros(X.ndim + 1)           #weight array for weights and bias(+1) 
    #np.random.shuffle(X)                    #can be used to randomize input array for efficiency
    iter = 10                                #setting max epoch size
    for _ in range(iter): 
        for data, label in zip(X, Y):
            prediction = train(data, label, weights)
            if prediction == 0:
                weights[1:] += (label - prediction) * data      #updating weight
                weights[0] += (label - prediction)              #updating bias
            else:
                continue
    return weights[1:] , weights[0]

#function to test perceptron
def test(X, label, weights):
    activation = np.dot(X, weights[0:2]) + weights[2]
    return label if activation*label > 0.0 else 0.0             #returning label for comparison

def perceptron_test(X,Y,w,b):
    weights = np.append(w, b)               #putting weights and bias together in 1 array
    labels = []                             #container for new labels
    correct = 0                             #counter for correctly classified labels
    #np.random.shuffle(X)                   #can be used to randomize input array for efficiency
    for data, label in zip(X, Y):
        labels.append(test(data, label, weights))
    
    for i in range(len(Y)):                 #calculate accuracy 
        if Y[i] == labels[i]:
            correct += 1  	
    return correct / float(len(Y))