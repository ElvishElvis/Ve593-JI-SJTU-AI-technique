from __future__ import division
import numpy as np
import math
import copy
import sys
import random
class Network(object):

    def __init__(self, sizes,activationFcns):
        """
        :param: sizes: a list containing the number of neurons in the respective layers of the network.
                See project description.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.activations=activationFcns

    def inference(self, x):
        """
        :param: x: input of ANN
        :return: the output of ANN with input x, a 1-D array
        """
        cur=None
        inputs=x
        for i in range(self.num_layers-1):
            cur=np.dot(self.weights[i],inputs)+self.biases[i]
            for l in range(len(cur)):
            #     if cur[l]>100:
            #         for o in range(len(cur)):
            #             cur[o]/=100
            # for k in range(len(cur)):
                cur[l][0]=self.activations[i+1](cur[l][0])
            inputs=cur
        return cur



    def training(self, trainData, T, n, alpha,lmbda,validationData):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        accu=0
        time=0
        while time<T:
            batches=[]
            for i in range(0, len(trainData), n):
                batches.append(trainData[i:i+n])
            for batch in batches:
                self.updateWeights(batch, lmbda,alpha)
            score=self.evaluate(validationData)
            if score<accu:
                print("restart")
                return False
            if score<=0.01:
                print("early stopping")
                return True
            accu=score
            print(score)
            if score>=0.93:
                print("Great!")
                return True
            time+=1
                

    def updateWeights(self, batch, lmbda,alpha):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """
        for i in range(len(batch)):
            (x,y)= batch[i]
            (newW,newB)=self.backprop(x,y)
            for l in range(len(newW)):
                # The L2 regularization
                # self.weights[l]=(1-alpha*lmbda/len(batch))*self.weights[l]-newW[l]*alpha
                # The L1 regularization
                # temp=copy.deepcopy(self.weights[l])
                # total=0
                # for k in range(len(temp)):
                #     total+=sum(temp[k])
                # if total>0:
                #     sign=1
                # else:
                #     sign=-1
                # self.weights[l]=self.weights[l]-alpha*lmbda*sign/len(batch)-newW[l]*alpha


                self.weights[l]-=newW[l]*alpha
                self.biases[l]-=newB[l]*alpha




    def backprop(self, x, y):
        if self.activations[2]==sigmoid:
            prime=sigmoid_prime
        elif self.activations[2]==tanh:
            prime=tanh_prime
        elif self.activations[2]==ReLU:
            prime=ReLU_prime
        elif self.activations[2]==LeakReLU:
            prime=LeakReLU_prime
        elif self.activations[2]==arctan:
            prime=arctan_prime

        nabla_b = [0]*2;
        nabla_w = [0]*2;
        input_list = [x] 
        temp = [] 
        for b, w in zip(self.biases, self.weights):
            result = np.dot(w, x)+b
            temp.append(result)
            x = self.activations[2](result)
            input_list.append(x)

        theta = dSquaredLoss(input_list[-1], y) * prime(temp[-1])

        nabla_b[1] = theta
        temp_list=np.array(list(map(list,zip(*input_list[-2]))))
        nabla_w[1] = np.dot(theta, temp_list)

        result = temp[-2]
        temp2 = prime(result)
        temp_list2=np.array(list(map(list,zip(*self.weights[-1]))))

        delta = np.dot(temp_list2, theta) * temp2
        nabla_b[0] = delta
        temp_list3=np.array(list(map(list,zip(*input_list[-3]))))
        nabla_w[0] = np.dot(delta, temp_list3)

        return (nabla_w, nabla_b)

    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """


        # result=np.array([])

        # for k in data:
        #     x=self.inference(k[0])
        #     y=k[1]
        #     if x[0]-y[0]<0.001 and x[1]-y[1]<0.0001:
        #         result=np.append(result,1)
        #     else:
        #         result=np.append(result,0)
        # return np.average(result)
        temp=[]
        for k in data:
            x=np.argmax(self.inference(k[0]))
            y=y=k[1]
            temp.append((x,y))
        return sum(int(x == y) for (x, y) in temp)/len(data)




# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    """
    :param a: vector of activations output from the network
    :param y: the corresponding correct label
    :return: the vector of partial derivatives of the squared loss with respect to the output activations
    """
    return a-y


    # return np.sum(y-a)


def sigmoid(z):
    """The sigmoid function"""
    try:
        z=1 / (1 + np.exp(-z))
    except OverflowError:
        pass
    #     print(z)
    #     sys.exit()
    return z


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
    if type(z)==np.float64:
        return math.atan(z)
    else:
        for l in range(len(z)):
            temp =z[l][0]
            num=max(temp,0)
            z[l][0]=num
        return z


def ReLU_prime(z):
    for l in range(len(z)):
        if z[l][0]>=0:
            z[l][0]=1
        else:
            z[l][0]=0
    return z
    


def tanh(z):
    return (1-np.exp(-2*z))/(1+np.exp(-2*z))


def tanh_prime(z):
    return 1-tanh(z)**2

def arctan(z):
    if type(z)==np.float64:
        return math.atan(z)
    else:
        for l in range(len(z)):
            z[l][0]=math.atan(z[l][0])
        return z

def arctan_prime(z):
    return 1/(z**2+1)

def LeakReLU(z):
    if type(z)==np.float64:
        return math.atan(z)
    else:
        for l in range(len(z)):
            temp =z[l][0]
            num=max(temp,0)+min(temp,0)*0.01
            z[l][0]=num
        return z


def LeakReLU_prime(z):
    for l in range(len(z)):
        if z[l][0]>=0:
            z[l][0]=1
        else:
            z[l][0]=0.01
    return z



