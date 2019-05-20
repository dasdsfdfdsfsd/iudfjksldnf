import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class DisplayResult(object):
    def __init__(self,rules,types):
        self.rules = ["RL", "IBS", "IBN", "LRS"]
        self.gtypes = ["", "-Malicious"]

        self.font = {
        #'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 25,
        }
        self.colors = ['r','b','g','y','k']
        self.linestyle = ['-',':','--']
        self.markers = ['o','D','*','x','>','<','v','^','s']


    def plot_curve(self,result,savefig=False):
        fig = plt.figure(figsize=(27,6))
        for net_index,data in enumerate(result):
            item = data[1:] 
            total_epoch = data.shape[-1] 

            plt.subplot(1,3,net_index+1)
            plt.xlim(-5, total_epoch+5)
            plt.ylim(-0.02,1.01)
            interval_y = 0.1
            interval_x = total_epoch/6
            plt.xticks(np.arange(0, total_epoch + interval_x, interval_x)) # 刻度
            plt.yticks(np.arange(0, 1 + interval_y, interval_y))
            plt.grid()
            plt.title(nets[net_index], self.font)
            plt.xlabel('Training Epoch', self.font)
            plt.ylabel('Final Cooperation Level', self.font)

            item = np.reshape(item,(-1,2,item.shape[-1])) 
            plt.tick_params(labelsize=25)

            y_axis = data[0] 
            plt.plot(range(len(y_axis)),y_axis,color='k', linestyle=self.linestyle[-1], lw=1.5)

            scatter_x = [i*100 for i in range(6)]+[599]
            scatter_y = [y_axis[i] for i in scatter_x]
            plt.scatter(scatter_x,scatter_y,marker=self.markers[-1],c='k',edgecolors='k',s=200)
            plt.plot(range(1),scatter_y[0],'--ks',label="DRL",lw=1.5,markerfacecolor='k',markersize=12 )
            for rule_index,CLevel in enumerate(item):
                rule = self.rules[rule_index]
                for gametype_index,value in enumerate(CLevel):
                    gametype = self.gtypes[gametype_index]
                    x_axis = range(len(y_axis))
                    plt.plot(x_axis,value,color=self.colors[rule_index], linestyle=self.linestyle[gametype_index], lw=1.5)

                    scatter_y = [value[i] for i in scatter_x]
                    plt.scatter(scatter_x,scatter_y,marker=self.markers[rule_index*2+gametype_index],c=self.colors[rule_index],edgecolors='',s=200)
                    plt.plot(range(1),scatter_y[0],self.linestyle[gametype_index]+self.colors[rule_index]+ self.markers[rule_index*2+gametype_index],label=rule+gametype,lw=1.5,markerfacecolor=self.colors[rule_index],markersize=12 )

        legend(bbox_to_anchor=(1.05, 1.23),ncol=9,fontsize=20)
        if savefig != False:
            plt.savefig('Experiment1.png', dpi=800, bbox_inches='tight')
            plt.savefig('Experiment1.eps', dpi=800, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def plot_bar(self,data,savefig=False,nets=None):
        labels = ["RL", "IBS", "IBN", "LRS"]
        hatchs = ['.', '*', 'x', 'O']
        # facecolors = ['r', 'b', 'g', 'y']
        plt.figure(figsize=(27,6))
        for i,item in enumerate(data):
            plt.subplot(1,3,i+1)
            plt.title(nets[i],self.font)
            plt.xlabel('Initial Cooperation Level',self.font)
            plt.ylabel('Final Cooperation Level',self.font)
            plt.ylim([0,1.02])
            plt.xticks([0.75,1.75,2.75],[0.2,0.5,0.7])
            plt.yticks([0.1*i for i in range(11)])
            plt.tick_params(labelsize=25)
            n = 3
            X = np.arange(n)+0.5

            for index,value in enumerate(item):
                plt.bar(X+0.15*index, value, alpha=1, width = 0.15, facecolor = 'white', edgecolor = self.colors[index],hatch=hatchs[index], label=labels[index], lw=3)
            plt.grid(axis='y')
        plt.legend(loc=1)
        legend(bbox_to_anchor=(-0.05, 1.25),ncol=4,fontsize=22)
        show()
        if savefig:
            plt.savefig('Experiment2.png', dpi=800, bbox_inches='tight',transparent=True)
            plt.savefig('Experiment2.eps', dpi=800, bbox_inches='tight',transparent=True)



    def plot_heatmap(self,results,networks,savefig=False):

        plt.figure(figsize=(9*len(results),6))
        grid = plt.GridSpec( 6,9*len(results),wspace=1,hspace=1)
        index = 0
        for data,network in zip(results,networks):
            if index ==0:
                ax = plt.subplot(grid[:6,0:8])

            elif index ==1:
                ax = plt.subplot(grid[:6,9:17])

            elif index == 2:
                ax = plt.subplot(grid[:6,18:])

            if index == 2:
                sns.set(font_scale=2.4)
                sns.heatmap(data, annot=False, vmax=1,vmin = 0,xticklabels= False, yticklabels= False,linewidths=.01, square=False, cmap=plt.cm.tab20c,
                       cbar_kws={'label': 'Level of cooperation'})
            else:
                sns.heatmap(data, annot=False, vmax=1,vmin = 0,xticklabels= False, yticklabels= False,linewidths=.01, square=False, cmap=plt.cm.tab20c,
                   cbar=False)

            ax.set_yticks(np.linspace(0,len(data),num=6), minor=False)
            ax.set_xticks(np.linspace(0,len(data),num=6), minor=False)

            ax.set_yticklabels(np.round(np.linspace(0.,0.5,num=6),2))
            ax.set_xticklabels(np.round(np.linspace(0,10,num=6),1), minor=False)
            plt.tick_params(labelsize=25)
            ax.invert_yaxis()
            plt.xlabel('Privacy budget',self.font)
            plt.ylabel(r'Proportion of malicious',self.font)
            plt.title(network,self.font)
            index += 1

        if savefig:
            plt.savefig("Experiment3.png",dpi=300,bbox_inches='tight')
            plt.savefig("Experiment3.eps",dpi=300,bbox_inches='tight')
        show()


