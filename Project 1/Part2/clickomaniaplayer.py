from clickomania import *
import random
def clickomaniaplayer(Gragh,initialState):
	
	frontier=[[Gragh]]
	while frontier:

		print 'running...'
		cur=frontier.pop(0)
		if cur[0].isEnd()==True:
			return (cur[::-1],)
		succ=cur[0].successors()
		temp=[]
		for k in range(len(succ)):
			temp.append([succ[k]])
		for t in range(len(temp)):
			temp[t].extend(cur)
		frontier.extend(temp)

def random_action(N,M,K):
	state=[]
	for i in range(N*M):
		ran=random.randint(0,K)
		state.append(ran)
	Map=Clickomania(N,M,K,state)
	print Map.row
	clickomaniaplayer(Map,state)
	print Map.state
	print Map.score

#If the score is 0, please run it again
random_action(5,5,3)