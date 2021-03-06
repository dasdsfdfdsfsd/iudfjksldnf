import random
import numpy as np
import sys

class ReinforcementLearning(object):
    def InitialQP(self,matrix):

        self.degree = np.sum(matrix,1) 
        self.QValue = {}
        self.NoisePi = {} 
        self.Pi = {} 
        for index,degree in enumerate(self.degree):
            self.QValue[index] = {}
            self.QValue[index] = np.concatenate([np.expand_dims(np.random.laplace(1.5, 0.2, int(degree)+1),1),np.expand_dims(np.random.laplace(1.5, 0.2, int(degree)+1,1),1).T
            self.Pi[index] = np.ones((2,int(degree)+1))*0.5

    
    def __init__(self,number,fractionC,fractionM,matrix,gameType,mu = 0.01):  
        self.count = 0       
            
        self.number=number
        self.fractionC = fractionC
        self.fractionM = fractionM
        self.gameType = gameType
        self.Action={0,1}
        self.neighbors = {} 
        self.HAction = {} 
        self.mu = mu 
        self.CurrentAgent = 0        
        self.Strategy = np.array([1] * round(number * fractionC) + [0] * round(number * (1 - fractionC)))
        random.shuffle(self.Strategy) 
 
        
        for i in range(self.number):
            self.neighbors[i] = list( np.reshape(np.argwhere(matrix[i]==1),(1,-1)) [0] ) # why reshape, reshape(x,1) means x rows and 1 column
            
            self.HAction[i] = np.ones(2)  
        self.InitialQP(matrix)
            
        if gameType in ["Malicious","Malicious_DP"]:
            self.MalicAgent = np.array(sorted(random.sample(range(number),int(fractionM*number))))
         
    
    def UpdateCurrentState(self,Agent = None):             
            index = self.CurrentAgent
            self.currentState = self.Strategy[self.neighbors[index]]      
            self.currentStateCode = list(self.currentState).count(1)    
    
    def ComputePayoff(self,T=1.2, R=1., P=0.1, S=0.):
        payoffs = np.zeros(self.number)
        differenceC = []    
        differenceD = [] 
        
        for agent in range(self.number):
            state = self.Strategy[self.neighbors[agent]] 
            niC = len(np.argwhere(state == 1))  
            niD = len(state) - niC            
            if self.Strategy[agent] == 1:
                payoff = niC * R + niD * S
            else:
                payoff = niC * T + niD * P
            payoffs[agent] = payoff
        self.payoffs = payoffs
    
    def NextState(self):
        nextStateCode = 0 #??
        for neig in self.neighbors[self.CurrentAgent]:
            HAction = self.HAction[neig] #??
            assert len(HAction) ! = 0
            HAction /= np.sum(HAction)
            randnum = np.random.rand(1)
            if randnum > HAction[0]:
                nextStateCode += 1
        return nextStateCode #neighbor 的状态
    
    def FakePayoff(self, Rewards, neighbor):
        neighbor_rewards = Rewards
        for index,neig in enumerate(neighbor):  
            if neig in self.MalicAgent:
                # if defrctor, then increase payoff
                if self.Strategy[self.CurrentAgent] == 0:
                    neighbor_rewards[index] *= 3.0
                else:
                    neighbor_rewards[index] *= 0.2
        return neighbor_rewards
    
    def DPNoise(self,data):         
        data_noise = np.exp(self.mu * data)
        data_noise /= np.sum(data_noise)
        return data_noise
    
    def UpdateQ(self,alpha=0.7,gamma=0.1):           
        self.UpdateCurrentState()        
        agent = self.CurrentAgent           
        state = self.currentStateCode
        action = self.Strategy[self.CurrentAgent]
        reward = self.payoffs[agent]        
        
        neighbor = self.neighbors[agent]
        neighbor_rewards = self.payoffs[neighbor]
        
        # fake payoff
        if self.gameType in ["Malicious","Malicious_DP"]:
            neighbor_rewards = self.FakePayoff(neighbor_rewards,neighbor)

        Rewards = np.zeros(len(neighbor_rewards) + 1)
        Rewards[:len(neighbor_rewards)] = neighbor_rewards #
        Rewards[-1] = reward  # 自己的reward 和neighbor的reward
        weight = Rewards/np.sum(Rewards) 
        if self.gameType == "Malicious_DP":
            weight = self.DPNoise(weight)
            
        term = np.sum(weight * Rewards)

        nextstate = self.NextState()
        pi = self.Pi[self.CurrentAgent][:,nextstate] # 某一个agent在某一个时刻下，的所有action的概率
        q = self.QValue[self.CurrentAgent][:,nextstate] 
        term2 = np.sum(pi * q)
        self.QValue[agent][action][state] =  (1-alpha) * self.QValue[agent][action][state] + alpha * (term + gamma * term2)

    def AverageReward(self):
        pi = self.Pi[self.CurrentAgent][:,self.currentStateCode]
        q = self.QValue[self.CurrentAgent][:,self.currentStateCode]    
        return np.sum(pi * q)
    
    def UpdateProbability(self,avgR):
        zeta =0.1
        action = self.Strategy[self.CurrentAgent]           
        q = self.QValue[self.CurrentAgent][action][self.currentStateCode] - avgR
        self.Pi[self.CurrentAgent][action][self.currentStateCode] += zeta * q 
    
    def NormalizePi(self):       
        arg = np.argwhere(self.Pi[self.CurrentAgent][:,self.currentStateCode] < 0)
        if len(arg) > 0:
            self.Pi[self.CurrentAgent][arg,self.currentStateCode] = 0.001
            self.Pi[self.CurrentAgent][1-arg,self.currentStateCode] = 0.999
        else:   
            totalPro = np.sum(self.Pi[self.CurrentAgent][:,self.currentStateCode])
            self.Pi[self.CurrentAgent][:,self.currentStateCode] /= totalPro
                    
    def NextAction(self):
        if self.gameType in ["Malicious","Malicious_DP"] and self.CurrentAgent in self.MalicAgent:           
            return random.sample([0,1],1)[0]
        probability = self.Pi[self.CurrentAgent][:,int(self.currentStateCode)]
        randnum = np.random.rand(1)
        if randnum < probability[0]:
            nextAction = 0
        else:
            nextAction = 1
        # Add selectied action in to history action 
        self.HAction[self.CurrentAgent][nextAction] += 1
        return nextAction

    # Cooperation Level without malicious in malicious game 
    def ComputeMCLevel(self):
        assert self.gameType != "Normal"
        self.CooperationLevel = 0
        for i in range(self.AgentNum):
            if i not in self.MalicAgent and self.Strategy[i] == 1:
                self.CooperationLevel += 1 
        self.CooperationLevel /= (self.AgentNum-len(self.MalicAgent))
    
    def ComputeCLevel(self):
        self.CooperationLevel = list(self.Strategy).count(1)/self.AgentNum    
    
    def play(self,rule,Epoch,index,string):
        CLevel = np.zeros(Epoch,dtype=np.float32)
        self.Epoch = Epoch # ？？
        for epoch in range(Epoch): 
            self.ComputePayoff()
            self.ComputeCLevel()
            self.explor_rate = epoch/Epoch # no use
            NextAction = [] #
            for i in range(self.number):
                self.CurrentAgent = i
                self.UpdateQ()
                avgR = self.AverageReward()
                self.UpdateProbability(avgR)
                self.NormalizePi()                
                NextAction.append(self.NextAction())
                self.HAction[i][NextAction[-1]] += 1
            self.Strategy = np.array(NextAction)
            CLevel[epoch] = self.CooperationLevel
            sys.stdout.write('\r>>Iteration: %d/%d \t Epoch: %d/%d \t Cooperation proportion: %2f' % (index+1,100,epoch+1,Epoch,CLevel[epoch]))  
            sys.stdout.flush()
            if string == 'InitialCL' :
                if epoch>100 and np.std(CLevel[epoch-30:epoch])<0.01:
                    return np.average(CLevel[epoch-30:epoch])
        if string == 'InitialCL' :
            return np.average(CLevel[epoch-30:epoch])
        return CLevel 