from __future__ import division
import numpy as np
import math
from decimal import Decimal

def MLEstimation(graph, data):
    gragh_flip=np.array(map(list,zip(*graph)))

    cptList=[]
    for i in range(len(gragh_flip)):
        variable=i
        parents=[]
        for k in range(len(gragh_flip[i])):
            if gragh_flip[i][k]==1:
                parents.append(k)
        cpt=MLEstimationVariable(variable, parents, data)
        cptList.append(cpt)
    return cptList
        
def MLEstimationVariable(variable, parents, data):
    data_flip=np.array(map(list,zip(*data)))
    cpt=[]
    possible_value=np.unique(data_flip[variable])
    num_occur=[]
    for i in possible_value:
        num_occur.append(np.sum(data_flip[variable]==i))
    if len(parents)==0:
        for k in num_occur:
            cpt.append([k/len(data_flip[variable])])
        return cpt
    else:
        possible_value_pa=[]
        for par in parents:
            #each possible value that the parents could have
            possible_value_pa.append(np.unique(data_flip[par]))
        situation_possible=parents_enumerate(possible_value_pa)
        # for each value parents get
        for situation in range(len(situation_possible)):
            cpt.append([])
            Nj=0
            situ=[0]*len(possible_value)
            # for each case that fit the situation
            for case in data:
                cur_int=0
                flag=True
                # if a case really fit
                while cur_int<len(parents):
                    if case[parents[cur_int]]==situation_possible[situation][cur_int]:
                        pass
                    else:
                        flag= False
                        break
                    cur_int+=1
                if flag== False:
                    continue
                else:
                    #it fit!
                    Nj+=1
                    for i in range(len(possible_value)):
                        if possible_value[i]==case[variable]:
                            situ[i]+=1       
            for b in range(len(situ)):
                cpt[situation].append(situ[b]/len(data_flip[variable]))
        return cpt

                      
def log_dd(num):
    digit=0
    while num >10:
        num/=10
        digit+=1
    return digit  


def K2Score_log(variable, parents, data):
    score=0
    print score
    possible_value=np.unique(data[variable])
    r=len(possible_value)
    if len(parents)==0:
        Nj=len(data[variable])
        num_occur=[]
        for i in possible_value:
            num_occur.append(np.sum(data[variable]==i))
        try:
            score+=log_dd(Decimal(math.factorial(r-1)))
            score-=log_dd(Decimal(math.factorial(Nj+r-1)))
        except ValueError:
            pass
        for b in range(len(num_occur)):
            score+=log_dd(Decimal(math.factorial(num_occur[b])))
        return score
    else:
        print score
        possible_value_pa=[]
        for par in parents:
            #each possible value that the parents could have
            possible_value_pa.append(np.unique(data[par]))
        situation_possible=parents_enumerate(possible_value_pa)
        data_flip=np.array(map(list,zip(*data)))
        # for each value parents get
        for situation in situation_possible:
            Nj=0
            situ=[0]*len(possible_value)
            # for each case that fit the situation
            for case in data_flip:
                cur_int=0
                flag=True
                # if a case really fit
                while cur_int<len(parents):
                    if case[parents[cur_int]]==situation[cur_int]:
                        pass
                    else:
                        flag= False
                        break
                    cur_int+=1
                if flag== False:
                    continue
                else:
                    #it fit!
                    Nj+=1
                    for i in range(len(possible_value)):
                        if possible_value[i]==case[variable]:
                            situ[i]+=1
            try:
                score+=log_dd(Decimal(math.factorial(r-1)))
                score-=log_dd(Decimal(math.factorial(Nj+r-1)))
            except ValueError:
                pass
            for b in range(len(situ)):
                score+=log_dd(math.factorial(situ[b]))
        return score
                            


def BICScore(variable, parents, data):
    score=0
    possible_value=np.unique(data[variable])
    r=len(possible_value)
    if len(parents)==0:
        Nj=len(data[variable])
        num_occur=[]
        for i in possible_value:
            num_occur.append(np.sum(data[variable]==i))

        for b in num_occur:
            score+=b*math.log(b/Nj)
        return 2*score
    else:
        possible_value_pa=[]
        for par in parents:
            #each possible value that the parents could have
            possible_value_pa.append(np.unique(data[par]))
        situation_possible=parents_enumerate(possible_value_pa)
        data_flip=np.array(map(list,zip(*data)))
        # for each value parents get
        for situation in situation_possible:
            Nj=0
            situ=[0]*len(possible_value)
            # for each case that fit the situation
            for case in data_flip:
                cur_int=0
                flag=True
                # if a case really fit
                while cur_int<len(parents):
                    if case[parents[cur_int]]==situation[cur_int]:
                        pass
                    else:
                        flag= False
                        break
                    cur_int+=1
                if flag== False:
                    continue
                else:
                    #it fit!
                    Nj+=1
                    for i in range(len(possible_value)):
                        if possible_value[i]==case[variable]:
                            situ[i]+=1
            for b in range(len(situ)):
                try:
                    score+=situ[b]*math.log(situ[b]/Nj)
                except (ValueError,ZeroDivisionError):
                    pass
        return 2*score-len(possible_value_pa)*(r-1)*math.log(data.shape[1])


def parents_enumerate(parent_list):
    num=1
    for lists in parent_list:
        num*= len(lists)
    final=[]
    for i in range(num):
        final.append([])
    for all_list in range(len(parent_list)):
        cur=parent_list[all_list]
        diff=len(cur)
        itera=num
        for i in range(0,all_list+1):
            itera/=len(parent_list[i]) 
        itera=int(itera)
        index=0
        numer=int(num/diff/itera)
        for c in range(numer):
            for l in cur:
                for k in range(itera):
                    if index==0:
                        final[k].append(l)
                    else:
                        final[k+itera*index].append(l)
                index+=1
    return final



def K2Algorithm(K, data, scoreFunction):
    data=np.array(map(list,zip(*data)))
    gragh=[]
    n=data.shape[0]
    parents= [-1]*n
    total_score=0
    for i in range(n):
        print "-----------------------------------"
        print "Current node "+ str(i)
       
        if i==0:
            gragh.append([0]*n)
            total_score+=scoreFunction(i,[],data)
            print "No parent for the first node"     
            print ""
            print ""
            continue

        parents[i]=[]
        gragh_list=[0]*n
        score=scoreFunction(i,parents[i],data)
        print "Old score = "+ str(score)
        print "-----------------------------------"
        countinue=True
        digit=0
        #to testify the parents of each node
        
        while countinue and len(parents[i])<K:
            data_list=[range(n)]
            for k in range(i,n):
                data_list[0].remove(k)
            for c in parents[i]:
                data_list[0].remove(c)
            if len(data_list[0])==0:
                print "reach the end of this node"
                break

            #the testify the n-th parents of each node
            max_score=float('-inf')
            max_parents=None
            #find the max score for each other node
            parents[i].append([-1])
            for b in data_list[0]:
                print "considering adding node" +str((i,b))
                parents[i][digit]=b
                score_temp=scoreFunction(i,parents[i],data)
                if score_temp>max_score:
                    max_score=score_temp
                    max_parents=b
                    print "temporary adding node"+str((i,b))
                    print "Temp score = "+ str(score_temp)
                    print "Current max score = "+ str(max_score)
                    print " "
                else:
                    print "No adding node"+str((i,b))
                    print "Temp score = "+ str(score_temp)
                    print "Current max score = "+ str(max_score)
                    print " "
            sNew=max_score
            if sNew>score:
                print "New score = "+ str(sNew)
                print "Old score = "+ str(score)
                print "adding node"+str((i,max_parents))
                print "-----"
                print "next node to go "
                print "-----"
                score=sNew
                parents[i][digit]=max_parents
                digit+=1
            else:
                parents[i][digit]=None
                print "New score = "+ str(sNew)
                print "Old score = "+ str(score)
                print "Not adding node"+str((i,max_parents))
                print "No result for parent " + str(digit+1)
                print ""
                countinue=False
        try:
            parents[i].remove(None)
        except ValueError:
            pass
        for par in parents[i]:    
            gragh_list[par]=1
        gragh.append(gragh_list)
        total_score+=score
        print ""
        print "Total score = " +str(total_score)
        print ""
        print ""
       
    gragh=np.array(map(list,zip(*gragh)))
    print "Final gragh "
    print str(gragh)
    print "Final Score"
    print total_score
    return [gragh, total_score]

            
def K2Score(variable, parents, data):
    score=1
    possible_value=np.unique(data[variable])
    r=len(possible_value)
    if len(parents)==0:
        Nj=len(data[variable])
        num_occur=[]
        for i in possible_value:
            num_occur.append(np.sum(data[variable]==i))
        score*=Decimal(math.factorial(r-1))/Decimal(math.factorial(Nj+r-1))
        for b in range(len(num_occur)):
            score*=Decimal(math.factorial(num_occur[b]))
        return score
    else:
        possible_value_pa=[]
        for par in parents:
            #each possible value that the parents could have
            possible_value_pa.append(np.unique(data[par]))
        situation_possible=parents_enumerate(possible_value_pa)
        data_flip=np.array(map(list,zip(*data)))
        # for each value parents get
        for situation in situation_possible:
            Nj=0
            situ=[0]*len(possible_value)
            # for each case that fit the situation
            for case in data_flip:
                cur_int=0
                flag=True
                # if a case really fit
                while cur_int<len(parents):
                    if case[parents[cur_int]]==situation[cur_int]:
                        pass
                    else:
                        flag= False
                        break
                    cur_int+=1
                if flag== False:
                    continue
                else:
                    #it fit!
                    Nj+=1
                    for i in range(len(possible_value)):
                        if possible_value[i]==case[variable]:
                            situ[i]+=1
            
            score*=Decimal(math.factorial(r-1))/Decimal(math.factorial(Nj+r-1))
            for b in range(len(situ)):
                if situ[b]==0:
                    continue
                score*=Decimal(math.factorial(situ[b]))
        return score
                            
    


