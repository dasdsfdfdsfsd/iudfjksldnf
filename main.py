import networkx as nx
import matplotlib.pyplot as plt
import community
from community import community_louvain
import random 
import numpy as np
import sys
import parser
from Network import Network
from Game import Game
from RLGame import ReinforcementLearning
from utils import *
import datetime
import argparse
from Plot import  DisplayResult

# Experiment1

def main():
     starttime = datetime.datetime.now()

     number = 1000
     degree = 4
     nets =  ['Homogeneous',"ScaleFree","Random" ]       
     rules = [ "RL-DP","BS", "BN","Redistribution"]
     gameTypes = ["Normal"] 
     runs = 1000       # number of initial network
     rounds = 600    # number of epoch 
     fractionM = 0.2
     fractionC = [0.2,0.5,0.7]
    
     Results = Recorder1(nets,rules,fractionC,rounds,runs)
     for net in nets:
         for rule in rules:
             print("\n Network:",net,"\t Rule:",rule)
             #100 simulation 
             for r in range(runs):
                 network = Network(number,degree,net)
                 matrix = network.generateNetworks()
                 for gtype in gameTypes :
                     for InitCL in fractionC:
                         if rule == 'RL-DP':
                             NGame = ReinforcementLearning(number,InitCL,fractionM,matrix,gtype)                         
                         else:
                             NGame = Game(number,rule,InitCL,fractionM,matrix,gtype) 
                         NCRate = NGame.play(rule,rounds,r,'InitialCL') 
                         Results.update(net,rule,NCRate,r,InitCL)                    
     data = Results.getResult()
     save_result(data)
     endtime = datetime.datetime.now()
     print((endtime - starttime).seconds)
     return data

if __name__ == '__main__':
    result = main()






# Experiment2

def main(args): 

    starttime = datetime.datetime.now()

    # parser = argparse.ArgumentParser(description='MultiAgent')
    # parser.add_argument('--number', type=int, default=1000)
    # parser.add_argument('--degree', type=int, default=4)
    # parser.add_argument('--nets', type=str, default='Homogeneous')
    # # parser.add_argument('-nets', '--list',nargs='+', action='store', dest='list', type=str,default= [Homogeneous,ScaleFree,Random],  required=True)
    # # parser.add_argument('--rules', type=str, default='RL-DP')
    # # parser.add_argument('--gameTypes', type=str, default='RL-DP')
    # parser.add_argument('--nets', '--list', action='append', required=True)
    # parser.add_argument('--rules', '--list', action='append', required=True)
    # parser.add_argument('--gameTypes', '--list', action='append', required=True)
    # parser.add_argument('--runs', type=int, default=20)
    # parser.add_argument('--rounds', type=int, default=600)
    # parser.add_argument('--fractionM', type=float, default=0.1)
    # parser.add_argument('--fractionC', type=float, default=0.5)


    Results = Recorder2(nets, rules, gameTypes, rounds, runs)
    for net in nets:
        EpochResults = []
        for rule in rules:
            print("\n Network:",net,"\t Rule:",rule)
            #gameTypes = ["Normal","Malicious","Malicious_DP"]            
            if rule == "RL-DP":
            	gameTypes = ["Normal","Malicious","Malicious_DP"]
            else:
            	gameTypes = ["Normal","Malicious"]			
            gameTypes = ["Malicious_DP"]
            for gtype in gameTypes:
                for r in range(runs):
                    network = Network(number,degree,net)
                    matrix = network.generateNetworks()
                    if rule == 'RL-DP':
                        NGame = ReinforcementLearning(number,fractionC,fractionM,matrix,gtype)
                    else:
                        NGame = Game(number, rule, fractionC, fractionM, matrix, gtype)
                    NCRate = NGame.play(rule, rounds, r, 'Epoch')
                    Results.update(net, rule, NCRate, r, gtype)
                data = Results.getResult()
                EpochResults.append(data[net][rule][gtype])
        np.savetxt('Experiment2_mu0.1_'+net+'.txt', np.array(EpochResults), fmt='%.2f')
        endtime = datetime.datetime.now()
        print("\n Running time:",(endtime - starttime).seconds)
    return Results, EpochResults
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='MultiAgent')  #not quite understand this 
    parser.add_argument('--number', type=int, default=1000)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--nets', type=str, default='Homogeneous')
    parser.add_argument('--nets', '--list', action='append', required=True)
    parser.add_argument('--rules', '--list', action='append', required=True)
    parser.add_argument('--gameTypes', '--list', action='append', required=True)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=600)
    parser.add_argument('--fractionM', type=float, default=0.1)
    parser.add_argument('--fractionC', type=float, default=0.5)
    args = parser.parse_args()
    result = main(args)





# Experiment3
def main():
    starttime = datetime.datetime.now()
    number = 1000
    degree = 4
    nets = 'Homogeneous'     
    rule =  "RL-DP" 
    gtype = "Malicious_DP" 
    runs = 1000       
    rounds = 600     
    fractionM = 0.2
    fractionC = 0.5
    EpochResults = []
    data = []
    for net in nets:
        for fractionM in np.linspace(0.,0.5,41):
            Row = []
            for noise in np.linspace(0.01,0.1,41):
                NCRate = 0
                for r in range(runs):
                    network = Network(number, degree, net)
                    matrix = network.generateNetworks()
                    if rule == 'RL-DP':
                        NGame = ReinforcementLearning(number, fractionC, fractionM, matrix, gtype, noise)
                    else:
                        NGame = Game(number, rule, fractionC, fractionM, matrix, gtype, noise)
                    NCRate += NGame.play(rule, rounds, r, 'InitialCL')
                Row.append(NCRate/runs)
            EpochResults.append(Row)
        np.savetxt('Experiment3_'+net+'.txt', np.array(EpochResults), fmt='%.2f')
        data.append(np.array(EpochResults))

    Display = DisplayResult()
    Display.plot_heatmap(data, nets, savefig=False)
    endtime = datetime.datetime.now()
    print("\n Running time:", (endtime - starttime).seconds)
    return EpochResults

if __name__ == '__main__':
    result = main()
