# encoding=utf-8
import matplotlib.pyplot as plt
from search import *
import datetime
import numpy as np

def convert(string):
	return float(str(int(string[0])*3600+int(string[2:4])*60+int(string[5:7]))+string[7:])

'''The running time for n=3 '''
Graph=nPuzzleGraph(3)
initialState=get_random(Graph,3)
time=[]
table={0:BFS(Graph,initialState),1:DFS(Graph,initialState),2:DLS(Graph,initialState,1000),\
3:IDS(Graph,initialState)}

for k in range(4):
	temp=[]
	for i in range(20):
		starttime = datetime.datetime.now()
		table[k]
		endtime = datetime.datetime.now()
		temp.append(convert(str(endtime-starttime)))
	time.append(np.average(temp))




'''The running time for n=3 '''
ValuedGraph=nPuzzleGraph_v(3)
initialState_value=get_random(Graph,3)
table_value={0:UCS(ValuedGraph,initialState_value),1:Astar(ValuedGraph,initialState_value,heuristic_1),\
2: MCTS(ValuedGraph,initialState_value,1000)}

for b in range(2):
	temp_value=[]
	for c in range(20):
		starttime = datetime.datetime.now()
		table_value[b]
		endtime = datetime.datetime.now()
		temp_value.append(convert(str(endtime-starttime)))
	time.append(np.average(temp_value))



''' Plot the computation time '''
names = ['5', '10', '15', '20', '25','22','16']
x = range(len(names))
y = time

plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'Running Time')


plt.legend()  

plt.xticks(x,['BFS', 'USC','DFS', 'DLS', 'IDS', 'Astar','MCTS'])

plt.xlabel(u"time(s)") 
plt.ylabel("Sorting Algorithm") 
plt.title("The computation time for search algorithm") 

plt.show()
