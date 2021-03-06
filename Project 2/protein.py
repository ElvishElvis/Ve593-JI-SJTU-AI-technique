from variableelimination import *
from structurelearning import *
import numpy as np
import copy

def topo_order(D_):
	gragh_flip=np.array(map(list,zip(*D_)))
	order=[ 4,1,3,2,0,5] 
	temp_list=[]
	final_list=[]
	for i in range(len(order)):
	    temp_list.append(gragh_flip[i])
	for d in order:
	    final_list.append(temp_list[d])
	final= np.array(map(list,zip(*final_list)))
	return final
	 

if __name__=='__main__':
	D = np.genfromtxt('protein.csv', delimiter=',')
	D_temp=D[1:]
	D_=topo_order(D_temp)
	# The total data is 19480
	index_for_test=19000
	D_train=D_[:index_for_test]
	D_test=D_[index_for_test:]

	gragh=K2Algorithm(2, D_train, BICScore)[0]
	cpt=MLEstimation(gragh,D_train)
	model_final=(gragh,cpt)

	index_interest=0

	accu_list=[]
	gragh_flip=np.array(map(list,zip(*D_))) 
	unique= np.unique(gragh_flip[0])
	model=copy.deepcopy(model_final)
	print "Still Running....."
	for case in D_test:
	    model_x=copy.deepcopy(model)
	    observe=[]
	    for i in range(1,6):
	        observe.append((i,case[i]))
	    prob=variable_elimination(0, observe, model_x)
	    
	    max_index=-1
	    max_prob=0
	    for each in range(len(prob)):
	        if prob[each]>max_prob:
	            max_index=each
	            max_prob=prob[each]
	    if unique[max_index]==case[0]:
	        accu_list.append(1)
	    else:
	        accu_list.append(0)
	#     print [unique[max_index],case[11]]
	print np.average(accu_list)
