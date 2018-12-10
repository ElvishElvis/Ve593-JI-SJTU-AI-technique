
class Clickomania:


    def __init__(self, N, M, K, state):
        self.row = N
        self.column = M
        self.color = K
        self.score = 0
        self.state = state
    #This make a copy from the state
    def clone(self):
        NewClickomania = Clickomania(self.row, self.column, self.color, self.state)
        NewClickomania.score = self.score
        return NewClickomania
    #This update the gragh
    def succ(self, position):
        nextstate = self.clone()
        Erasable = [position]
        self.CanErase(position, self.state[position], Erasable)

        if len(Erasable) >= 2:
            for i in Erasable:
                nextstate.state[i] = 0

        self.score += (len(Erasable)-1)**2

        for x in list(range(0, self.column)):
            for y in list(range(1, self.row)):
                if (self.state[x+y*self.column] == 0):
                    for z in list(range(y, 0, -1)):
                        self.state[x+z*self.column] = self.state[x+(z-1)*self.column]
                    self.state[x] = 0

        Column=[]
        for i in list(range(0, self.column)):
            flag = 0
            for j in list(range(0, self.row)):
                if (self.state[i+j*self.column] != 0):
                    flag = 1
                    break
            if (flag == 0):
                Column.append(i)

        while (Column != []):
            k = Column.pop(0)
            if (k != self.column-1):
                for r in list(range(0, self.row)):
                    for s in list(range(1, self.column-k)):
                        if ((k+r*self.column+s) % self.column != self.column-1):
                            self.state[k+r*self.column+s] = self.state[k+r*self.column+s+1]
            for t in list(range(0, self.row)):
                self.state[self.column-1+t*self.column] = 0
            for m in range(len(Column)):
                Column[m] -= 1

        return nextstate


    def successors(self):
        succs = []
        for i in list(range(0, self.column*self.row)):
            succs.append(self.succ(i))
        return succs
    #This determine whether a state reach the end state
    def isEnd(self):
        AllErased = True
        adjacent = []
        for i in list(range(0, self.row*self.column)):
            if (self.state[i] != 0):
                AllErased = False
                if (self.CanErase(i, self.state[i], adjacent) == []):
                    return True
        if AllErased == True:
            return True
        return False

    #This is a function that can determine whether a block can be erased
    def CanErase(self, place, k, adjacent):
        if place>self.column:
            if ( place >= self.column and  (place-self.column not in adjacent) and self.state[place-self.column] == k):
                adjacent.append(place-self.column)
                self.CanErase(place-self.column, k, adjacent)
        if place<self.row*self.column:
            if ( place % self.column != self.column-1 and  (place+1 not in adjacent) and self.state[place+1] == k):
                adjacent.append(place+1)
                self.CanErase(place+1, k, adjacent)
        if place<(self.row-1)*self.column:
            if ( place < self.column*(self.row-1) and  (place+self.column not in adjacent) and self.state[place+self.column] == k):
                adjacent.append(place+self.column)
                self.CanErase(place+self.column, k, adjacent)
        if place>0:
            if ( place % self.column != 0 and (place-1 not in adjacent) and self.state[place-1] == k ):
                adjacent.append(place-1)
                self.CanErase(place-1, k, adjacent)
        return adjacent
    #This reduce the penalty to the final score
    def Cost(self, state):
        NotErased
        for i in list(range(0, self.row*self.column)):
            if (state.value[i] != 0):
                NotErased += 1
        return (NotErased-1)**2







