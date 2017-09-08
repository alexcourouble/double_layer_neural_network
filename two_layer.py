"""
This is my implementation of the TWO hidden layer NN with mini batch

Alex Courouble

INF8225
"""


import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import numpy as np

class Neural_Network(object):
    
    
    def __init__(self):
        #defining the layer sizes
        self.inputLayerSize = 784
        self.hiddenLayer1Size = 200
        self.hiddenLayer2Size = 200
        self.outputLayerSize = 10
        
        #defining the weight matrices
        self.w1 = np.random.randn(self.inputLayerSize,self.hiddenLayer1Size)
        self.w2 = np.random.randn(self.hiddenLayer1Size,self.hiddenLayer2Size)
        self.w3 = np.random.randn(self.hiddenLayer2Size,self.outputLayerSize)
        
        #self.a = .1
        
        
        
    """ forward pass method"""
    def forward(self, x):
        #Propagate inputs though network
        self.a1 = np.dot(x, self.w1)
        self.h1 = self.activation(self.a1)
        self.a2 = np.dot(self.h1, self.w2)
        self.h2 = self.activation(self.a2)
        #output
        self.outputAct = np.dot(self.h2, self.w3)
        self.output = self.activation(self.outputAct)
        
        return self.output
        
        
        
    """Activation function using sigmoid"""   
    def activation(self, z):
        return 1/(1+np.exp(-z))
        
    def activationPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    
    
    """Softmax function"""
    def softmax(self, z):
        newZ = []
        for i in z:
            exp_i = np.exp(i)
            sum = exp_i.sum()
            for j in range(len(exp_i)):
                exp_i[j] = exp_i[j]/sum
            newZ.append(exp_i)
        return newZ
        
        
        
    def costFunction(self, y, x):
        yHat = NN.forward(x)
        theList = []
        for i in range(len(yHat)):
            for j in range(len(yHat[i])):
                theList.append((y[i][j]-yHat[i][j])**2)
        J = 0.5 * np.average(theList)
        return J



    """ this is the backpropagation"""
    def costFunctionPrime(self, y, x):
        yHat = NN.forward(x)
        newArr = []
        for i in range(len(y)):
            newVec = []
            for j in range(len(y[i])):
                newVec.append(y[i][j]-yHat[i][j])
            newArr.append(newVec)
        newArr = np.asarray(newArr)
        
        D4 = np.multiply(-(newArr), NN.activationPrime(self.outputAct))
        dJdW3 = np.dot(self.h2.T, D4)
        
        D3 = np.dot(D4, self.w3.T) * NN.activationPrime(self.a2)
        dJdW2 = np.dot(self.h1.T, D3)
        
        D2 = np.dot(D3, self.w2.T) * NN.activationPrime(self.a1)
        dJdW1 = np.dot(x.T, D2)
        
        return dJdW1, dJdW2, dJdW3
        
"""split into mini batch"""
def createBatches(matrix,numberOfBatches):
    return np.split(matrix, numberOfBatches)
    

""" creating the one-hot function for y"""
def createTable(array):
    newTable = []
    count = 0
    for i in array:
        newTable.append([0,0,0,0,0,0,0,0,0,0])
        newTable[count][i] = 1
        # print newTable[count]
        count += 1
    return newTable
        
        
if __name__ == "__main__":
    print "TWO HIDDEN LAYER"
    # opening the data and creating the sets
    with open('mnist.pkl', 'rb') as f:
    	train_set, valid_set, test_set = pickle.load(f)  
    train_x, train_y = train_set
    test_x, test_y = test_set
    valid_x, valid_y = valid_set
    #creating yTrain 
    yTrain = createTable(train_y)
    yValid = createTable(valid_y)
    yTest = createTable(test_y)
    #convertiing everything to numpy array
    xTrain = np.asarray(train_x)
    yTrain = np.asarray(yTrain)
    xValid = np.asarray(valid_x)
    yValid = np.asarray(yValid)
    xTest = np.asarray(test_x)
    yTest = np.asarray(yTest)
    
    costTrain = []
    costValid = []
    NN = Neural_Network()
    # print "train",NN.costFunction(yTrain, xTrain)
#     print "valid",NN.costFunction(yValid, xValid)
    
    
    numberOfBatches = 100
    xTrainBatches = createBatches(xTrain,numberOfBatches)
    yTrainBatches = createBatches(yTrain,numberOfBatches)
    xValidBatches = createBatches(xValid,numberOfBatches)
    yValidBatches = createBatches(yValid,numberOfBatches)
    
    for i in range(30):
        
        for j in range(numberOfBatches):
            
            dJdW1, dJdW2, dJdW3 = NN.costFunctionPrime(yTrainBatches[j], xTrainBatches[j])
            scalar = .00001

            NN.w1 = NN.w1 - scalar*dJdW1
            NN.w2 = NN.w2 - scalar*dJdW2
            NN.w3 = NN.w3 - scalar*dJdW3
        
            costTrain.append(NN.costFunction(yTrainBatches[j], xTrainBatches[j]))
            costValid.append(NN.costFunction(yValidBatches[j], xValidBatches[j]))
    
    print "train", costTrain
    print "valid", costValid
    print "test",NN.costFunction(yTest, xTest)
    
    
    plt.plot(costTrain,"r",costValid,"g")
    plt.ylabel('Error rate')
    plt.show()
        