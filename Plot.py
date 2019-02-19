import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_heatmap(data,network,savefig=False):
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 15,
    }

    # xmajorLocator   = MultipleLocator(0.1) #将x主刻度标签设置为20的倍数
    # xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式
    # xminorLocator   = MultipleLocator(0.02) #将x轴次刻度标签设置为5的倍数

    # ymajorLocator   = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
    # ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
    # yminorLocator   = MultipleLocator(0.02) #将此y轴次刻度标签设置为0.1的倍数


    fig, ax = plt.subplots(figsize = (9,6))

    sns.heatmap(data, annot=False, vmax=1.,vmin = 0., xticklabels= True, yticklabels= False,linewidths=.01, square=False, cmap=plt.cm.CMRmap,
               cbar_kws={'label': 'Level of cooperation'})


    ax.set_yticks(np.linspace(0,41.0,num=11), minor=False)
    ax.set_xticks(np.linspace(0.0,41.0,num=11), minor=False)

    ax.set_yticklabels(np.round(np.linspace(0.00,0.5,num=11),2))
    ax.set_xticklabels(np.round(np.linspace(0.01,0.1,num=11),2), minor=False)
    ax.invert_yaxis()
    plt.xlabel('\n Level of noise',font1) 
    plt.ylabel(r'Proportion of malicious',font1) 
    plt.title(network,font2) 
    # plt.title("Homogeneous\n"+r'$[\theta=1.0]$',font2) 

    # plt.title(r'Microstrain [$mu epsilon$]')
    if savefig:
        fig.savefig("Heatmap"+network+".png",dpi=300,bbox_inches='tight')
    show()


result = np.loadtxt("Experiment3_Homogeneous.txt") 
# plot_heatmap(result,"Homogeneous",savefig=True)

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(0, 10, 0.33)
Y = np.arange(0, 10, 0.33)
X, Y = np.meshgrid(X, Y)  
Z = result
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

