import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
'''
fig, ax =plt.subplots(2,1,constrained_layout=True, figsize=(10, 12))

################ plot bnm ############
bnm0 = list(np.load(open("../outputs/bnm0.0.npy","rb")))
bnm0_5 = list(np.load(open("../outputs/bnm0.5.npy","rb")))
bnm0_ = []; bnm0_5_ = []
for i in range(len(bnm0)):
    if bnm0[i] > 650: bnm0_.append(bnm0[i])
    if bnm0_5[i] > 650: bnm0_5_.append(bnm0_5[i])
step = [i for i in range(len(bnm0_))]
print(len(bnm0_5),len(bnm0))

print(len(bnm0_+bnm0_5_),len(step + step), len(["kappa = 0.0"]* len(bnm0) + ["kappa = 0.5"]* len(bnm0_5)))
df = pd.DataFrame({"matrix_rank":bnm0_ + bnm0_5_,"step":step + step, "method": ["kappa = 0.0"]* len(bnm0_) + ["kappa = 0.5"]* len(bnm0_5_)})
sns.lineplot(data=df, x="step", y="matrix_rank",hue = "method",ax=ax[0])
ax[0].set_title("(a). TOD37")

#######################
dd_bnm0 = list(np.load(open("../outputs/daily_bnm0.0.npy","rb")))[:1750]
dd_bnm0_5 = list(np.load(open("../outputs/daily_bnm0.5.npy","rb")))[:1750]
dd_bnm0_ = []; dd_bnm0_5_ = []
for i in range(len(dd_bnm0)):
    if dd_bnm0[i] > 60: dd_bnm0_.append(dd_bnm0[i])
    if dd_bnm0_5[i] > 60: dd_bnm0_5_.append(dd_bnm0_5[i])
step = [i for i in range(len(dd_bnm0_))]
print(len(dd_bnm0_5_),len(dd_bnm0_))

print(len(dd_bnm0_+dd_bnm0_5_),len(step + step), len(["kappa = 0.0"]* len(dd_bnm0) + ["kappa = 0.5"]* len(dd_bnm0_5)))
df = pd.DataFrame({"matrix_rank":dd_bnm0_ + dd_bnm0_5_,"step":step + step, "method": ["kappa = 0.0"]* len(dd_bnm0_) + ["kappa = 0.5"]* len(dd_bnm0_5_)})
sns.lineplot(data=df, x="step", y="matrix_rank",hue = "method",ax=ax[1])
ax[1].set_title("(b). DailyDialog")
plt.savefig("plot.png")
'''

import random
fig, ax =plt.subplots(2,1,constrained_layout=True, figsize=(10, 12))
bnm0_4_pool = [22.96, 21.17, 25.87, 19.41, 19.01, 21.38, 14.1, 19.91, 19.8, 20.37, 25.68, 19.76, 34.53, 15.18, 26.46, 21.23, 27.22, 23.44, 22.54, 22.99, 18.62, 43.65, 26.4, 20.59, 21.72, 26.31, 25.28, 17.48, 14.56, 31.22, 25.92, 29.36, 19.95, 27.14, 15.36, 30.49, 36.09]
mixup = [21.94, 21.18, 25.51, 18.2, 19.57, 21.62, 14.23, 19.72, 19.13, 19.37, 25.1, 20.21, 33.04, 14.63, 26.36, 20.71, 25.57, 24.35, 22.37, 23.03, 19.43, 45.59, 25.47, 19.24, 22.39, 27.05, 25.83, 17.3, 14.52, 31.32, 26.67, 29.13, 19.63, 23.76, 14.79, 30.08, 33.81]
task_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37] 
ewc = [20.8, 21.3, 22.83, 15.79, 14.13, 20.52, 14.36, 17.73, 18.74, 19.08, 24.38, 24.6, 39.02, 14.79, 29.95, 20.36, 30.78, 21.54, 18.52, 17.1, 18.0, 27.47, 22.32, 17.2, 23.95, 26.32, 21.03, 16.0, 14.53, 31.8, 24.53, 28.9, 19.05, 12.03, 17.53, 29.57, 36.39]
replay = [21.18, 20.61, 24.74, 16.89, 18.04, 17.99, 11.83, 16.97, 19.38, 16.18, 23.04, 18.29, 32.34, 16.48, 25.9, 19.72, 25.89, 21.6, 21.61, 22.56, 19.52, 43.14, 26.53, 18.06, 22.5, 27.51, 22.98, 16.94, 13.09, 30.45, 25.83, 28.53, 19.78, 21.91, 14.7, 28.24, 32.39]
lamol = [20.93, 21.46, 22.44, 16.94, 14.99, 20.94, 14.21, 18.21, 18.56, 20.2, 24.25, 24.5, 39.69, 18.15, 31.01, 19.85, 30.65, 21.57, 18.19, 18.3, 18.06, 27.47, 22.27, 17.51, 23.36, 27.01, 20.12, 15.99, 14.25, 32.27, 24.5, 28.91, 19.56, 10.64, 16.93, 29.59, 36.72]
finetune = [21.37, 18.27, 21.52, 18.27, 14.9, 20.43, 11.66, 16.17, 17.74, 16.98, 21.1, 19.27, 36.11, 16.35, 27.95, 18.75, 24.07, 20.83, 18.43, 19.31, 15.52, 24.79, 19.99, 15.34, 20.68, 24.53, 19.58, 16.29, 12.12, 29.08, 22.34, 27.22, 19.58, 11.78, 13.12, 28.31, 29.74]
print(len(task_id)*5,len(bnm0_4_pool+replay+mixup+finetune+lamol),len(["bnm"]*37+["replay"]*37+["mixup"]*37+["multi"]*37+["lamol"]*37))
df = pd.DataFrame({"task_id":task_id*6,"BLEU":bnm0_4_pool+replay+mixup+finetune+ewc+lamol,"method":["bnm"]*37+["replay"]*37+["mixup"]*37+["finetune"]*37+["ewc"]*37+["lamol"]*37})
print(df)
ax[0].set_title("(a). TOD37")

sns.lineplot(data=df,x='task_id',y='BLEU',hue = "method",ax = ax[0])
ax[0].set_xlabel("task through time")
#ax1.set_xticks([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37])


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
plt.savefig("plot1.png")

