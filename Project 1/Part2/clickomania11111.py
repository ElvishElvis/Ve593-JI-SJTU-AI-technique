class State:
   """State for nPuzzle graph that stores the position of the blank."""
   
   def __init__(self, v):
      self.value = v

   def clone(self):
      return State(list(self.value))

   def __repr__(self):
      """Converts an instance in string.

         Useful for debug or with print(state)."""
      return str(self.value)

   def __eq__(self, other):
      """Overrides the default implementation of the equality operator."""
      if isinstance(other, State):
         return self.value == other.value
      elif other==None:
         return False
      return NotImplemented

   def __hash__(self):
      """Returns a hash of an instance.

         Useful when storing in some data structures."""
      return hash(str(self.value))

class Clickomania:
    """simple Clickomania"""

    def __init__(self, N, M, K, initialState):
        self.row = N
        self.column = M
        self.color = K
        self.score = 0
        self.initialstate = initialState

    def clone(self):
        nextClickomania = Clickomania(self.row, self.column, self.color, self.initialstate)
        nextClickomania.score = self.score
        return nextClickomania

    def succ(self, position):
        print "position is"
        print position
        nextstate = self.clone()
        print nextstate.initialstate
        adjacent=[position]

        """eliminate a bunch of adjacent block with same color and gain score"""
        self.findAdjacent(position, self.initialstate[position], adjacent)
        if len(adjacent)>1:
            for i in adjacent:
                nextstate.initialstate[i] = 0
                print "nextstate is "
                print nextstate.initialstate

        """add score in this elimination"""
        self.score += (len(adjacent)-1)**2

        """check whether a column is empty"""
        emptyColumn=[]
        for i in list(range(0, self.column)):
            flag = 0
            for j in list(range(0, self.row)):
                if (self.initialstate[i+j*self.column] != 0):
                    """not empty in that column"""
                    flag = 1
                    break
            if (flag == 0):
                """the column is empty"""
                emptyColumn.append(i)
        print "emptyColumn is"
        print emptyColumn

        """move left"""
        while (emptyColumn != []):
            print "in while"
            k = emptyColumn.pop(0)
            if (k != self.column-1):
                for r in list(range(0, self.row)):
                    for s in list(range(1, self.column-k)):
                        print "Hugh"
                        print k+r*self.column+s

                        print (k+r*self.column+s) % self.column
                        print self.column-1
                        if ((k+r*self.column+s) % self.column != self.column-1):
                            print k+r*self.column+s
                            self.initialstate[k+r*self.column+s] = self.initialstate[k+r*self.column+s+1]
            for t in list(range(0, self.row)):
                """set the rightmost column to be 0"""
                self.initialstate[self.column-1+t*self.column] = 0
            for m in range(len(emptyColumn)):
                """update the emptyColumn list by minus 1"""
                emptyColumn[m] -= 1

        print "after while"
        print nextstate.initialstate

        """move down"""
        for x in list(range(0, self.column)):
            for y in list(range(1, self.row)):
                if (self.initialstate[x+y*self.column] == 0):
                    print "problem here"
                    print list(range(y, 0, -1))
                    for z in list(range(y, 0, -1)):
                        print x+z*self.column
                        self.initialstate[x+z*self.column] = self.initialstate[x+(z-1)*self.column]
                    """set the upmost row to be 0"""
                    self.initialstate[x] = 0
        print "after move down"
        print nextstate.initialstate
        return nextstate


    def successors(self):
        succs = []
        for position in list(range(0, self.row*self.column)):
            succs.append(self.succ(position))
        return succs

    def isEnd(self):
        flag = 0
        adjacent = []
        for position in list(range(0, self.row*self.column)):
            if (self.initialstate[position] != 0):
                flag = 1
                if (self.findAdjacent( position, self.initialstate[position], adjacent) == []):
                    return True
        if flag == 0:
            return True
        return False


    def findAdjacent(self, position, k, adjacent):
        """position is from 0 to N*M-1 """
        if position>0:
            if (position % self.column != 0 and self.initialstate[position-1] == k and (position-1 not in adjacent)):
                """Not in the leftmost column"""
                adjacent.append(position-1)
                self.findAdjacent(position-1, k, adjacent)
        if position<self.row*self.column:
            if (position % self.column != self.column-1 and self.initialstate[position+1] == k and (position+1 not in adjacent)):
                """Not in the rightmost column"""
                adjacent.append(position+1)
                self.findAdjacent(position+1, k, adjacent)
        if position>self.column:
            if (position >= self.column and self.initialstate[position-self.column] == k and (position-self.column not in adjacent)):
                """Not in the uppermost row"""
                adjacent.append(position-self.column)
                self.findAdjacent(position-self.column, k, adjacent)
        if position<(self.row-1)*self.column:
            if (position < self.column*(self.row-1) and self.initialstate[position+self.column] == k and (position+self.column not in adjacent)):
                """Not in the base row"""
                adjacent.append(position+self.column)
                self.findAdjacent(position+self.column, k, adjacent)
        print "adjacent are "
        print adjacent
        return adjacent

    def penalty(self, state):
        blockLeft = 0
        for i in list(range(0, self.row*self.column)):
            if (state.value[i] != 0):
                blockLeft += 1
        return (blockLeft-1)**2







