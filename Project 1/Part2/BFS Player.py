from clickomania import *
import random
def clickomaniaplayer(Gragh,initialState):
	frontier = [[Gragh]]
	print frontier[0][0].initialstate

	while frontier:
		# set up the upper bound of searching
		print 'running...'
		next = []
		for i in frontier:
			if (i[0].isEnd() == True):
				return (i[::-1], )
			succ = i[0].successors()
			temp = []

			for k in range(len(succ)):
				temp.append([succ[k]])
			for t in range(len(temp)):
				temp[t].extend(i)
			print frontier
			next.extend(temp)
		print frontier[0][0].initialstate
		frontier = next


def random_action(N,M,K):
	state=[]
	for i in range(N*M):
		ran=random.randint(0,K)
		state.append(ran)
	Map=Clickomania(N,M,K,state)
	print Map.row
	print clickomaniaplayer(Map,state)
	print Map.initialstate
	print Map.score


random_action(5,5,3)