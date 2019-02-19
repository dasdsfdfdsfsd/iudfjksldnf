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

def plot_curve(data,epoch,save_path=False):
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
    }
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 20,
    }
    colors = ['b','r','g','k']
    linestyle = [':','-','--']
    for net in data.keys():
        title = net
        total_epoch = epoch
        dpi = 80  
        width, height = 2000, 500
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)

        plt.xlim(-0.02, total_epoch+0.02)
        plt.ylim(-0.02, 1.02)
        interval_y = 0.1
        interval_x = 20
        plt.xticks(np.arange(0, total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 1 + interval_y, interval_y))
        plt.grid()
        plt.title(title, font2)
        plt.xlabel('the training epoch', font1)
        plt.ylabel('level of cooperation', font1)
  
        for index,rule in enumerate(data[net].keys()):
            if rule != 'RL':
                keyList = list(data[net][rule].keys())[:-1]
#                 print(keyList,type(keyList))
            else:
                keyList = data[net][rule].keys() 
            for index2,gType in enumerate(keyList):
                y_axis = data[net][rule][gType] 
                x_axis = range(len(y_axis))
                plt.plot(x_axis,y_axis,color=colors[index], linestyle=linestyle[index2], label=rule+" "+gType+" Game", lw=2)        
                plt.legend()
    
        if save_path != False:           
            fig.savefig(net+'.png', dpi=800, bbox_inches='tight')
#             print('---- save figure {} into {}'.format(title, save_path))
        plt.show()
        plt.close(fig)    