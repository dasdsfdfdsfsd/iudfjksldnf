import networkx as nx 
import matplotlib.pyplot as plt
import community
from community import community_louvain
import random 
import numpy as np
import sys

class Recorder2(object):
    """ store results """
    # different game type for RL need codes
    def __init__(self,nets,rules,Types,epoch,runs):
        self.epoch = epoch
        self.runs = runs
        self.coperation_rate = {}
        for key1 in nets:
            self.coperation_rate[key1] = {}
            for key2 in rules:       
                self.coperation_rate[key1][key2] = {}
                for gameType in Types:
                    self.coperation_rate[key1][key2][gameType] =  np.zeros(epoch,dtype=np.float32)                 
        
    def update(self,net,rule,CRate,r,gmaeType):
        self.coperation_rate[net][rule][gmaeType] += CRate
        if r == self.runs -1:
            self.coperation_rate[net][rule][gmaeType] = np.around(self.coperation_rate[net][rule][gmaeType]/self.runs,2)
            
    def getResult(self):
        return self.coperation_rate  

class Recorder1(object):
    """ store results """
    # 3 nets|4 methods|X:
    def __init__(self,nets,rules,initCL,epoch,runs):
        self.epoch = epoch
        self.runs = runs
        self.coperation_rate = {}
        for key1 in nets:
            self.coperation_rate[key1] = {}
            for key2 in rules:       
                self.coperation_rate[key1][key2] = {}
                for cl in initCL:
                    self.coperation_rate[key1][key2][cl] =  0.0     
        
    def update(self,net,rule,CRate,r,cl):
        self.coperation_rate[net][rule][cl] += CRate      
        if r == self.runs -1:
            self.coperation_rate[net][rule][cl] = round(self.coperation_rate[net][rule][cl]/self.runs,3)    
    def getResult(self):
        return self.coperation_rate  

def save_result(data,filename):
    fileObject = open(filename, 'w')
    for net in data.keys():
        fileObject.write(net)
        for rule in data[net].keys():
            result=list(data[net][rule].values())
            fileObject.write('\n ')
            fileObject.write(str(result))
        fileObject.write('\n \n')
    fileObject.close()    

def save_result2(data,filename):
    results = []
    fileObject = open(filename, 'w')
    for net in data.keys():
        for rule in data[net].keys():
            for gtype in data[net][rule].keys():
                result=list(data[net][rule][gtype])
                results.append(result)
    np.savetxt(filename,np.array(result))
