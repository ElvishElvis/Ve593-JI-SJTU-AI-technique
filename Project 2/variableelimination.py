from structurelearning import *

def the_parents(index,gragh_flip):
    direct_par=[]
    for i in range(len(gragh_flip[index])):
        if gragh_flip[index][i]==1:
            direct_par.append(i)
    return direct_par


def variable_elimination(index, observations, model):
    gragh= model[0]
    cpt=model[1]
    final_index=0
    gragh_flip=np.array(map(list,zip(*model[0])))

    paren=the_parents(index,gragh_flip)

    par_list=[]
    for par in paren:
        for i in observations:
            if i[0]==par:
                par_list.append(i)
    #unique number of each parent
    index_of=[]
    temp=[]
    for k in par_list:
        pa=the_parents(k[0],gragh_flip)
        
        if len(pa)==0:
            index_of.append(len(cpt[k[0]]))
        else:
            index_of.append(len(cpt[k[0]][0]))
    #get the probability of each direct parent 
    for c in range(len(par_list)):
        num=1
        for l in range(c+1,len(par_list)):
            num*=index_of[l]
        temp.append(par_list[c][1]*num)

    fina1_index=sum(temp)
    return cpt[index][int(final_index)]






