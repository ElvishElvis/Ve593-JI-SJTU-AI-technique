from Networks import *
from database_loader import *
import random
import datetime
import math
starttime = datetime.datetime.now()



training_data=list(load_data()[0])[:10000]
validation_data=list(load_data()[1])[:10000]
test_data=list(load_data()[2])


#My own design dataset

# training_data=[]
# validation_data=[]
# for i in range(1000):
#   a=([[random.random()],[random.random()]],[[random.random()],[random.random()]])
#   training_data.append(a)
#   c=([[random.random()],[random.random()]],[[random.random()],[random.random()]])
#   validation_data.append(c)







Net=Network([784, 30, 10], [None, sigmoid, sigmoid])
Batch_size=10
number_iter=20
learning_rate=0.5
lmbda=math.e**(-3)

Flag=False
time=0
while Flag==False:
    Flag=Net.training(training_data, number_iter, Batch_size, learning_rate,lmbda,validation_data)
    if Flag==False:
        time+=1
    if time>=5:
        break
    
print("The training accuracy is :" )
print(Net.evaluate(validation_data))
print("The testing accuracy is :" )
print(Net.evaluate(test_data))
endtime = datetime.datetime.now()
print("The running time is :")
print (str(endtime - starttime)+"seconds")