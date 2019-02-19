import networkx as nx 
import matplotlib.pyplot as plt
import community
from community import community_louvain
import random 
import numpy as np
import sys

class Network(object):
    def __init__(self,number,degree,net):
        """
        number: number of agents
        degree: averager degree 
        net: ScaleFree, Homogeneous or Random
        """
        self.number = number
        self.degree = degree
        self.net = net        
    
    def adjacencyMatrix(self,edges):
        """ Adjacency matrix """
        Matrix = np.zeros([self.number,self.number])
        for item in edges:
            Matrix[item[0]][item[1]] = 1.
            Matrix[item[1]][item[0]] = 1.
        return np.array(Matrix)


    def generateNetworks(self,DegreeDistribution=False):
        if self.net == 'ScaleFree':
            net= nx.random_graphs.barabasi_albert_graph(self.number,self.degree)   
        elif self.net == 'Homogeneous':
            net=  nx.random_graphs.random_regular_graph(self.degree,self.number) 
        elif self.net =='Random':
            net = nx.random_graphs.erdos_renyi_graph(self.number,self.degree/self.number) 
        else:
            print("Erro! Please give a right name of networks: Homogeneous, Random, ScaleFree")
     
    #         if DegreeDistribution == True:
#             x,y = self.getDegree(net)
#             self.Bar(x,y)
        edges = list(net.edges())
        matrix = self.adjacencyMatrix(edges)
        
        # make the zero degree nodes connected
        if self.net =='Random':
            matrix = self.processMatrix(matrix)
        
        if DegreeDistribution == True:
            x,y = self.getDegree(np.sum(matrix,1))
            self.Bar(x,y)
        return matrix
    
    def processMatrix(self,matrix):
        NewMatrix = matrix.copy()
        arg = np.argwhere(np.sum(matrix,1)==0)
        for item in arg:
            randomNum = random.randint(0,self.number-1)
            for i in range(3):
                while randomNum==item[0]:
                    randomNum = random.randint(0,self.number-1)
                NewMatrix[item[0]][randomNum] = 1
                NewMatrix[randomNum][item[0]] = 1
        assert len(np.argwhere(np.sum(NewMatrix,1)==0))==0
        return NewMatrix
    
    def getDegree(self,degrees):
        degree = {}
        for item in degrees:
            if item not in degree.keys():
                degree[item] = 1
            else:
                degree[item] += 1
        degree = dict(sorted(degree.items(), key=lambda e:e[0]))
        x = list(degree.keys())
        y = list(degree.values())

        return x,y
    
    def Bar(self,x,y):    
        lenx = round(max(x)-min(x))
        leny = round(max(y)-min(y))
        lenx = lenx/max(lenx,leny)
        leny = leny/max(lenx,leny)

    #     plt.figure(1, (leny*8, lenx*8))
        plt.figure(1, (8, 6))
        plt.bar(x, y, alpha=0.9, width = 0.8, facecolor = 'blue', edgecolor = 'white', label='one', lw=1)
        plt.xlim(min(x)-0.5,max(x)+0.5)
        plt.ylim(min(y),max(y)+0.5)

        plt.xticks(x, x)
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.grid(axis='y')
        plt.show()