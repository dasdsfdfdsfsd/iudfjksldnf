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
# from Plot import  DisplayResult


def Experiment1(args):
     starttime = datetime.datetime.now()
     Results = Recorder1(args.nets,args.rules, args.fractionC, args.rounds, args.runs)
     for net in args.nets:
         for rule in args.rules:
             print("\n Network:",net,"\t Rule:",rule)
             #100 simulation 
             for r in range(args.runs):
                 network = Network(args.number, args.degree, net)
                 matrix = network.generateNetworks()
                 for gtype in args.gameTypes :
                     for InitCL in args.fractionC:
                         if rule == 'RL-DP':
                             NGame = ReinforcementLearning(args.number,InitCL, args.fractionM,matrix,gtype)
                         else:
                             NGame = Game(args.number,rule,InitCL, args.fractionM,matrix,gtype)
                         NCRate = NGame.play(rule,args.rounds,r,'InitialCL')
                         Results.update(net,rule,NCRate,r,InitCL)                    
     data = Results.getResult()
     save_result(data)
     endtime = datetime.datetime.now()
     print((endtime - starttime).seconds)
     return data



def Experiment2(args):
    starttime = datetime.datetime.now()
    Results = Recorder2(args.nets, args.rules, args.gameTypes, args.rounds, args.runs)
    for net in args.nets:
        EpochResults = []
        for rule in args.rules:
            print("\n Network:", net,"\t Rule:",rule)
            #gameTypes = ["Normal","Malicious","Malicious_DP"]            
            if rule == "RL-DP":
                args.gameTypes = ["Normal","Malicious","Malicious_DP"]
            else:
                args.gameTypes = ["Normal","Malicious"]
            gameTypes = ["Malicious_DP"]
            for gtype in args.gameTypes:
                for r in range(args.runs):
                    for fracC in args.fractionC:
                        network = Network(args.number, args.degree, net)
                        matrix = network.generateNetworks()
                        if rule == 'RL-DP':
                            NGame = ReinforcementLearning(args.number, fracC, args.fractionM, matrix, gtype)
                        else:
                            NGame = Game(args.number, args.rule, fracC, args.fractionM, matrix, gtype)
                        NCRate = NGame.play(rule, args.rounds, r, 'Epoch')
                        Results.update(net, rule, NCRate, r, gtype)
                data = Results.getResult()
                EpochResults.append(data[net][rule][gtype])
        np.savetxt('Experiment2_mu0.01_'+net+'.txt', np.array(EpochResults), fmt='%.2f')
        endtime = datetime.datetime.now()
        print("\n Running time:",(endtime - starttime).seconds)
    return Results, EpochResults



def Experiment3(args):
    starttime = datetime.datetime.now()
    rule = args.rules[0]
    gtype = args.gameTypes[0]
    fractionC = args.fractionC[0]
    EpochResults = []
    data = []
    for net in args.nets:
        for fractionM in np.linspace(0.,0.5,41):
            Row = []
            for noise in np.linspace(0.01,0.1,41):
                NCRate = 0
                for r in range(args.runs):
                    network = Network(args.number, args.degree, net)
                    matrix = network.generateNetworks()
                    if rule == 'RL-DP':
                        NGame = ReinforcementLearning(args.number, fractionC, fractionM, matrix, gtype, noise)
                    else:
                        NGame = Game(args.number, rule, fractionC, fractionM, matrix, gtype, noise)
                    NCRate += NGame.play(rule, args.rounds, r, 'InitialCL')
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
    parser = argparse.ArgumentParser(description='MultiAgent')
    parser.add_argument('--number', type=int, default=1000)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('-n', '--nets', nargs='+', required=True)
    parser.add_argument('-r', '--rules', nargs='+', required=True)
    parser.add_argument('-g', '--gameTypes', nargs='+', required=True)
    parser.add_argument('-fc', '--fractionC', nargs='+', type=float, required=True)

    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=600)
    parser.add_argument('--fractionM', type=float, default=0.1)
    # parser.add_argument('--fractionC', type=float, default=0.5)
    parser.add_argument('--experiment', type=int, default=2)

    args = parser.parse_args()

    if args.experiment == 1:
        result = Experiment1(args)
    if args.experiment == 2:
        result = Experiment2(args)
    if args.experiment == 3:
        result = Experiment3(args)


'''
# First Experiment

data =  []
nets = ['Homogeneous',"Random","Scale-free" ]
rules = ["RL", "IBS", "IBN", "LRS"]
gtypes = ["", "-Malicious"]
Display = DisplayResult(rules,gtypes)

for net in nets:
    results = np.loadtxt("E:\Jupyter\\2018\DP\Experiment\Experiment2_"+net+".txt")
    results[[0, 2], :] = results[[2, 0], :]
    results[[1,2], :] = results[[2, 1], :]
    results[0] = np.loadtxt("E:\Jupyter\\2018\DP\Experiment\Experiment2_mu0.01_"+net+".txt")
    data.append(results)
Display.plot_curve(data, savefig = False,nets=nets)


# Second Experiment
result1 = np.array([ 0.945, 0.966,0.963,
0.001, 0.292, 0.834,
0.095, 0.478, 0.628,                    
0.418, 0.383, 0.402,

0.824,  0.828,  0.834,
0.007, 0.378, 0.987,
0.192, 0.831, 0.831,
0.609, 0.625, 0.683,
                    
0.612, 0.636,  0.686,
0.0, 0.253, 0.957,                    
0.099, 0.299, 0.498,
0.1, 0.4, 0.4])
result1 = np.reshape(result1,(3,-1,3))

# Display.plot_bar(result1, savefig = False,nets=nets)


# data =  []
# for net in nets:
#     results = np.loadtxt("E:\Jupyter\\2018\DP\Experiment\Experiment3_"+net+".txt")
#     data.append(results)
# Display.plot_heatmap(data,nets,savefig=False)
'''
