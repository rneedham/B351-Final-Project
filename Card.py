import numpy as np
import tensorflow as tf
import numpy as np
import heapq
import framework

def typeHelper(v):
    if v == b"Minion":
        return 0
    elif v == b"Spell":
        return 1
    else:
        return 2
    pass
 
def rarityHelper(v):
    if v == b"Free":
       return 0
    elif v == b"Common":
        return 1
    elif v == b"Rare":
        return 2
    elif v == b"Epic":
        return 3
    else:
        return 4
    pass

def classHelper(v):
    if v == b"Neutral":
        return 0
    elif v == b"Druid":
        return 1
    elif v == b"Hunter":
        return 2
    elif v == b"Mage":
        return 3
    elif v == b"Paladin":
        return 4
    elif v == b"Priest":
        return 5
    elif v == b"Rogue":
        return 6
    elif v == b"Shaman":
        return 7
    elif v == b"Warlock":
        return 8
    else:
        return 9
    pass

def tribeHelper(v):
    if v == b"None":
        return -1
    elif v == b"General":
        return 0
    elif v == b"Beast":
        return 1
    elif v == b"Demon":
        return 2
    elif v == b"Dragon":
        return 3
    elif v == b"Elemental":
        return 4
    elif v == b"Mech":
        return 5
    elif v == b"Murloc":
        return 6
    elif v == b"Pirate":
        return 7
    elif v == b"Totem":
        return 8
    else:
        return 9
    pass

class Card:
    
    def __init__(self, fileName):
        converters = { 5: rarityHelper }
        self.x = np.loadtxt(fileName, delimiter=',', converters = converters, skiprows=1, usecols=(2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
        
        self.expandedX = []
        for i in range(len(self.x)):
            card = []
            for j in range(len(self.x[i])): #creates inner array of an exploded card
                for k in range(len(self.x[i])):
                    card.append(self.x[i][j] * self.x[i][k])
            self.expandedX.append(card) #appends exploded cards into the new array of exploded cards
        
        self.y = np.loadtxt(fileName, delimiter=',', skiprows=1, usecols=(49, 50))



