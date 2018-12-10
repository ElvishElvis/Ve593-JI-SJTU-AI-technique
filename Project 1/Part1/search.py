from testgraphs import * 
from testgraphs_v import *
import random
import datetime
import numpy as np
import math

def BFS(Gragh,initialState):

	frontier=[[initialState]]
	seen=set()

	while frontier:
		#set up the upper bound of searching
		print 'running...'
		next=[]
		for i in frontier:
			#print i
			if i[0] in seen:
				continue
			else:
				seen.add(i[0])
			if Gragh.isGoal(i[0])==True:
				return (i[::-1],len(i)-1)
			succ=Gragh.successors(i[0])
			temp=[]
			for k in range(len(succ)):
				temp.append([succ[k]])
			for t in range(len(temp)):
				temp[t].extend(i)
			
			next.extend(temp)
		frontier=next

def DFS(Gragh,initialState):
	
	frontier=[[initialState]]
	seen=set()
	while frontier:
		#set up the upper bound of searching
		print 'running...'
		cur=frontier.pop(0)
		if cur[0] in seen:
			continue
		else:
			seen.add(cur[0])
		
		if Gragh.isGoal(cur[0])==True:
			return (cur[::-1],len(cur)-1)

		succ=Gragh.successors(cur[0])[::-1]
		temp=[]
		for k in range(len(succ)):
			temp.append([succ[k]])
		for t in range(len(temp)):
			temp[t].extend(cur)
		frontier.extend(temp)

def DLS(Gragh,initialState,depthlimit):
	#Gragh.n=3
	
	frontier=[[initialState,1]]
	seen=set()
	level=0
	while frontier:
		#set up the upper bound of searching
		print 'running...'	
		cur=frontier.pop(0)
		
		if cur[1]>=depthlimit:
			continue
		if cur[0] in seen:
			continue
		else:
			seen.add(cur[0])
		if Gragh.isGoal(cur[0])==True:
			result=cur[::-1]
			for b in result:
				if type(b)==type(1):
					result.remove(b)
			return (result,len(result)-1)
		succ=Gragh.successors(cur[0])[::-1]
		temp=[]
		for k in range(len(succ)):
			temp.append([succ[k],cur[1]+1])
		for t in range(len(temp)):
			temp[t].extend(cur)
		frontier.extend(temp)



		
	


def IDS(Gragh,initialState):
	def IDS_DLS(Gragh,initialState,depthlimit):
	
		frontier=[[initialState,1]]
		seen=set()
		level=0
		while frontier:
			#set up the upper bound of searching
			print 'running...'
			
			cur=frontier.pop(0)
			
			if cur[1]>=depthlimit:
				continue
			if cur[0] in seen:
				continue
			else:
				seen.add(cur[0])
			if Gragh.isGoal(cur[0])==True:
				result=cur[::-1]
				for b in result:
					if type(b)==type(1):
						result.remove(b)
				return (result,len(result)-1)
			succ=Gragh.successors(cur[0])[::-1]
			temp=[]
			for k in range(len(succ)):
				temp.append([succ[k],cur[1]+1])
			for t in range(len(temp)):
				temp[t].extend(cur)
			frontier.extend(temp)

	level=1
	while True:
		result=IDS_DLS(Gragh,initialState,level)
	
		if result==None:
			pass
		else:
			print level
			return result
		level+=1

def UCS(ValuedGraph,initialState):

	frontier=[[(0,initialState)]]
	seen=set()
	seen.add(initialState)
	time=0
	next=[]
	while frontier:
		# set up the upper bound of searching
		if time>=10000:
			return 'Not found'
		time+=1
		print 'running...'
		
		for i in frontier:
			succ=ValuedGraph.successors(i[0][1])

			for k in range(len(succ)):
				succ[k]=(succ[k][0]+i[0][0],succ[k][1])


			temp=[]
			for k in range(len(succ)):
				temp.append([succ[k]])
			for t in range(len(temp)):
				temp[t].extend(i)
				
			next.extend(temp)
			
		next=sorted(next,key=lambda x:x[0][0])
		
		while True:
			result = next[0]
			if result[0][1] in seen:
				next.pop(0)
			else:
				break
			if len(next)==0:
				return None
		
		next.remove(result)
		frontier.append(result)
		
		frontier.pop(0)
		seen.add(result[0][1])
		
		if ValuedGraph.isGoal(result[0][1])==True:
			return (result[::-1],result[0][0])


def Astar(ValuedGraph, initialState, heuristic):
	
	frontier=[[(0,initialState,0)]]
	seen=set()
	seen.add(initialState)
	time=0
	next=[]
	while frontier:
		# set up the upper bound of searching
		if time>=100000:
			return 'Not found'
		time+=1
		print 'running...'
		
		for i in frontier:
			succ=ValuedGraph.successors(i[0][1])
			for k in range(len(succ)):
				succ[k]=(succ[k][0]+i[0][0],succ[k][1],heuristic(succ[k][1]))
			temp=[]
			for b in range(len(succ)):
				temp.append([succ[b]])
			for t in range(len(temp)):
				temp[t].extend(i)
				
			next.extend(temp)
			
		next=sorted(next,key=lambda x:x[0][2])
		

		while True:
			result = next[0]
			#print result[0][1]
			if result[0][1] in seen:
				next.pop(0)
			else:
				break
			if len(next)==0:
				return None

		next.remove(result)
		frontier.append(result)
		
		frontier.pop(0)
		seen.add(result[0][1])
		
		if ValuedGraph.isGoal(result[0][1])==True:
			for c in range(len(result)):
				result[c]=(lambda x: (x[0],x[1])) (result[c])
			return (result[::-1],result[0][0])

#heuristic function for nPuzzleGraph
def heuristic_1(state):

	def difference(a,b):
		matrix=[1, 2, 3, 4, 5, 6, 7, 8,0]
		a_index=matrix.index(a)
		b_index=matrix.index(b)
		return abs((a_index/3)-(b_index/3))+abs((a_index%3)-(b_index%3))
	state=state.value
	sum_all=0
	for i in range(8):
		a=difference(state[i],i+1)
		sum_all+=a
	sum_all+=difference(state[8],0)
	return sum_all

#heuristic function for SimpleGraph
def heuristic_2(state):
	return 5-state



def MCTS(ValuedGraph,state,bound):
	frontier=[state]
	seen={}
	time=bound
	path=[(0,state)]
	had_seen={state}

	

	def sub_MCTS(ValuedGraph,state,bound):
	#set up the upper bound of searching
		print 'running...'
		next=ValuedGraph.successors(state)
		final=[]
		for k1 in next:
			#print k1
			bound_t=bound
			k=k1[1]
			current=k
			UCB={current:heuristic_2(current)}
			if current not in seen:
				seen[current]=1
			else:
				seen[current]+=1
			successors=ValuedGraph.successors(current)

			for i in range(50):	
				temp=[k]
				aver=[]
				while bound_t>0:
					if len(successors)==1:
						cur=successors[0]
					else:

						cur=successors[random.randint(0,len(successors)-1)][1]

					if cur not in seen:
						seen[cur]=1
					else:
						seen[cur]+=1


					if cur not in UCB:
						UCB[cur]=heuristic_2(cur)/seen[cur]+(math.log(2*seen[current])/seen[cur])**0.5
					else:
						pass
					temp.append(cur)
					bound_t-=1
				
				aver.append(UCB[temp[-1]])
			final.append((np.average(aver),k))
			
		final=sorted(final,key=lambda x:x[0])
		while True:
			try:
				now=final[0]
				ok=now[1]
			except IndexError:
				print 'Please run again'
				return None
			if ok in had_seen:
				final.pop(0)
			else:
				break
		
		return final[0]


	while ValuedGraph.isGoal(frontier[-1]) !=True and time>0:
		state=sub_MCTS(ValuedGraph,frontier[-1],bound)
		if state==None:
			return None

		path.append(state)
		frontier.append(state[1])
		had_seen.add(state[1])
		
		time-=1
	return (path,len(path))



def get_random(Gragh,n):
	initialstate=[]
	for i in range(n**2-1):
		initialstate.append(i+1)
	initialstate.append(0)
	cur=State(initialstate)
	for i in range(500):
		state=Gragh.successors(cur)
		num=random.randint(0,len(state)-1)
		cur=state[num]
	return cur






def value_run():
	starttime = datetime.datetime.now()

	"""This is the test act for SimpleGraph with value """
	ValuedGraph=SimpleValuedGraph()
	initialState=random.randint(0,4)
	#print UCS(ValuedGraph,initialState)



	"""This is the test act for nPuzzleGraph with value """
	ValuedGraph=nPuzzleGraph_v(3)
	Gragh=nPuzzleGraph(3)
	initialState=get_random(Gragh,3)
	#print UCS(ValuedGraph,initialState)
	# print MCTS(ValuedGraph,initialState,1000)
	print Astar(ValuedGraph,initialState,heuristic_1)

	endtime = datetime.datetime.now()
	print 'The running time is '+ str(endtime - starttime)+' Hour:Minute:Second:Microsecond'

def value():
	starttime = datetime.datetime.now()

	"""This is the test act for SimpleGraph with value """
	#Graph=SimpleGraph()
	#initialState=random.randint(0,4)
	# print BFS(Graph,initialState)
	# print DFS(Graph,initialState)
	# print DLS(Graph,initialState,3)
	# print IDS(Graph,initialState)



	"""This is the test act for nPuzzleGraph"""
	Graph=nPuzzleGraph(3)
	initialState=get_random(Graph,3)
	#print initialState
	#print BFS(Graph,initialState)
	#print DFS(Graph,initialState)
	#print DLS(Graph,initialState,50)
	#print IDS(Graph,initialState)


	endtime = datetime.datetime.now()
	print 'The running time is '+ str(endtime - starttime)+' Hour:Minute:Second:Microsecond'


