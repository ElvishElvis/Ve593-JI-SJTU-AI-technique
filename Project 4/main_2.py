from Conv import *
import numpy as np

import time
import struct
from glob import glob
import sys
import pickle


def load(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    datas = data['data'].reshape((10000,3,32,32)).swapaxes(1,3)
    labels = data['labels']
    return datas,labels

images, labels = load('data_batch_1')




batch_size = 200
num_batch = 10
learning_rate = 0.01
lmbda = 0.04
mode="ReLu"


conv1 = Conv2D([batch_size, 32, 32, 3], 12, 5, 1)
activate1 = activation(conv1.output_shape,mode)
pool1 = MaxPooling(activate1.shape_output)
conv2 = Conv2D(pool1.shape_output, 24, 3, 1)
activate2 = activation(conv2.output_shape,mode)
pool2 = MaxPooling(activate2.shape_output)
fc = FC(pool2.shape_output, 10)
sf = Softmax(fc.shape_output)


print("Number of trainning data : " + str(images.shape[0]))
print("Batch Size : " + str(batch_size))
print("Learning rate : "+str(learning_rate ))
print("L2 Regularization lambda : "+ str(lmbda))
print( " ")

flag=True
epoch=0
epoch_num=50
while (epoch<epoch_num and flag):

    
    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    train_acc = 0
    train_loss = 0
    i=0
    while( i <num_batch*batch_size and flag):
        img = images[i * batch_size:(i + 1) * batch_size]
        label = labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = activate1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = activate2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        batch_loss += sf.cross_entropy(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.result[j]) == label[j]:
                batch_acc += 1

        deltas = conv1.backward(activate1.backward(pool1.backward(conv2.backward(\
            activate2.backward(pool2.backward(fc.backward(\
                sf.backward(), learning_rate, lmbda))),learning_rate, lmbda))),learning_rate, lmbda)

        if i % 100 == 0: 
            print("epoch  "+str(epoch)+"  batch  "+str(i)+"  acc  "+str(batch_acc / float( batch_size))+"  loss  "+str( batch_loss / batch_size))
            
        if batch_acc / float( batch_size)>=0.95:
            flag=False
            print("epoch  "+str(epoch)+"  batch  "+str(i)+"  acc  "+str(batch_acc / float( batch_size))+"  loss  "+str( batch_loss / batch_size))
            break
        batch_loss = 0
        batch_acc = 0
        i+=1
print( " ")
print("The trainning accuracy is: "+ str(batch_acc / float( batch_size)))


b=0
count_=0
for b in range(1000):
    img = test_images[b * batch_size:(b + 1) * batch_size]
    label = test_labels[b * batch_size:(b + 1) * batch_size]
    conv1_out = activate1.forward(conv1.forward(img))
    pool1_out = pool1.forward(conv1_out)
    conv2_out = activate2.forward(conv2.forward(pool1_out))
    pool2_out = pool2.forward(conv2_out)
    fc_out = fc.forward(pool2_out)
    val_loss += sf.cross_entropy(fc_out, np.array(label))

    for j in range(batch_size):
        count_+=1
        if np.argmax(sf.result[j]) == label[j]:
            val_acc += 1
print(" ")
print ( "Validation accuracy: "+str(val_acc / float(count_))+   "  Validation loss: "+ str( val_loss / count_))

