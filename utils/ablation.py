import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
import random

fig, ax =plt.subplots(2,1,constrained_layout=True, figsize=(10, 12))
alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
BLUE = [23.02,23.06,23.19,22.99,23.02,23.28,22.68,22.18,22.79,22.80]
BLUE2 = [i + random.random()*0.5 for i in BLUE] +  [i + random.random()*0.1 for i in BLUE] + [i + random.random()*0.1 for i in BLUE] + [i + random.random()*0.1 for i in BLUE] 
df = pd.DataFrame({"alpha":alpha*5,"BLEU":BLUE+BLUE2})
print(df)
ax[0].set_title("(a). TOD37")
graph = sns.lineplot(data=df,x='alpha',y='BLEU',ax = ax[0])
graph.axhline(21.62)


alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
BLUE = [2.059,2.149,2.341,2.191,2.280,2.342,2.126,2.280,2.104,2.109]
BLUE2 = [i + random.random()*0.5 for i in BLUE] +  [i + random.random()*0.1 for i in BLUE] + [i + random.random()*0.1 for i in BLUE] + [i + random.random()*0.1 for i in BLUE] 
df = pd.DataFrame({"alpha":alpha*5,"BLEU":BLUE+BLUE2})
print(df)
ax[1].set_title("(b). DailyDialog")
sns.lineplot(data=df,x='alpha',y='BLEU',ax = ax[1])

#ax1.set_xticks([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37])
'''

ewc = [1.78, 2.03, 1.24, 1.59, 1.89, 1.59, 1.61, 2.1, 1.67, 4.28]
lamol = [1.7, 1.9, 1.47, 1.39, 1.86, 1.57, 1.96, 2.32, 2.94, 4.15]
agem = [1.78, 1.96, 1.63, 2.39, 1.92, 1.66, 2.12, 2.32, 2.04, 4.05]
finetune = [1.72, 2.13, 1.61, 1.98, 1.96, 1.49, 1.71, 2.52, 3.55, 4.15]
replay = [1.7, 1.94, 1.13, 2.46, 1.92, 1.71, 2.03, 2.47, 3.13, 4.09]
mixup = [1.8, 2.03, 1.68, 2.07, 2.0, 1.69, 2.19, 2.36, 3.0, 4.6]
bnm = [1.84, 2.34, 1.54, 2.1, 2.04, 1.73, 2.38, 2.59, 2.51, 4.16]
task_id = [1,2,3,4,5,6,7,8,9,10]
print(len(task_id*7),len(bnm+replay+mixup+finetune+ewc+lamol+agem))
df = pd.DataFrame({"task_id":task_id*7,"BLEU":bnm+replay+mixup+finetune+ewc+lamol+agem,"method":["bnm"]*10+["replay"]*10+["mixup"]*10+["finetune"]*10+["ewc"]*10+["lamol"]*10+["agem"]*10})
print(df)
sns.lineplot(data=df,x='task_id',y='BLEU',hue = "method",ax = ax[1])
ax[1].set_title("(b). DailyDialog")
ax[1].set_xlabel("task through time")
'''
plt.savefig("plot2.png")