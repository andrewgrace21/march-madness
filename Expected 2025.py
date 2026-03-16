import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

T = 20000
N = 5

teams = ['Auburn', 'Alabama St.', 'Louisville', 'Creighton', 'Michigan', 'UC San Diego', 'Texas A&M', 'Yale', 'Ole Miss', 'North Carolina', 'Iowa St.', 'Lipscomb', 'Marquette', 'New Mexico', 'Mcihigan St.', 'Bryant', 'Florida', 'Norfolk St.', 'UConn', 'Oklahoma', 'Memphis', 'Colorado St.', 'Maryland', 'Grand Canyon', 'Missouri', 'Drake', 'Texas Tech', 'UNC Wilmington', 'Kansas', 'Arkansas', 'St. Johns', 'Omaha', 'Duke', 'Mount St. Marys', 'Mississippi St.', 'Baylor', 'Oregon', 'Liberty', 'Arizona', 'Akron', 'BYU', 'VCU', 'Wisconsin', 'Montana', 'Saint Marys', 'Vanderbilt', 'Alabama', 'Robert Morris', 'Houston', 'SIU Edwardsville', 'Gonzaga', 'Georgia', 'Clemson', 'McNeese', 'Purdue', 'High Point', 'Illinois', 'Xavier', 'Kentucky', 'Troy', 'UCLA', 'Utah St.', 'Tennessee', 'Wofford']
NetRtg = [19.63, -7.94, 10.57, 12.52, 16.57, -1.44, 16.52, -1.85, 16.98, 12.57, 12.83, -4.15, 12.78, 7.33, 14.72, -6.41, 16.15, -6.42, 11.08, 14.9, 6.51, 6.55, 12.51, -3.03, 13.34, 1.23, 12.19, -4.06, 15.79, 15.21, 9.16, -1.91, 11.51, -6.61, 15.11, 16.72, 15.19, 1.06, 16.93, -3.34, 12.66, 1.32, 14.53, -1.74, 6.61, 13.38, 20.14, -5.15, 15.63, -8.19, 8.09, 14.18, 8.32, -.61, 16.76, -4.82, 17.14, 10.29, 18.23, -1.81, 14.09, 6.11, 16.80, .03]
df = pd.read_csv("pred_pop_men_2025.csv")
ppl_pick = np.array(df[['round1','round2','round3','round4','round5','round6']])
seeds = np.array([1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]*4)
full_seed = np.reshape(df[['seed']], 64)
history = np.array([[0.65, 0.13, 0.10, 0.05, 0.00, 0.03, 0.03, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], [1.03, 0.33, 0.28, 0.10, 0.10, 0.05, 0.03, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], [1.65, 0.80, 0.42, 0.38, 0.22, 0.08, 0.08, 0.15, 0.05, 0.02, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00], [2.68, 1.80, 1.02, 0.62, 0.30, 0.42, 0.25, 0.22, 0.12, 0.22, 0.25, 0.05, 0.00, 0.00, 0.02, 0.00], [3.40, 2.58, 2.10, 1.92, 1.38, 1.18, 0.72, 0.40, 0.20, 0.60, 0.68, 0.55, 0.15, 0.05, 0.10, 0.00], [3.95, 3.72, 3.42, 3.18, 2.58, 2.45, 2.45, 1.92, 2.08, 1.55, 1.55, 1.42, 0.82, 0.58, 0.28, 0.05]])


gen0 = []
x = np.array(NetRtg.copy())
for round in range(6):
    gen0.append([])
    for i in range(2**(5-round)):
        win = int(x[2*i]<x[2*i+1])
        gen0[-1].append(win)
    x = x[np.add(gen0[-1], range(0,len(x),2))]

def win_prob(adjEM_A, adjEM_B, seed_A, seed_B, poss=67, sigma=11):
    mu = (adjEM_A - adjEM_B) * poss / 100 + (seed_B**.0625 - seed_A**.0625) * 48
    return norm.cdf(mu / sigma)

prob_mtx = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        prob_mtx[i][j] = win_prob(NetRtg[i],NetRtg[j],full_seed[i],full_seed[j])

tourns = np.zeros((T,63))
ops = np.zeros((T,N,63))
for t in range(T):
    temp = []
    x = np.arange(64)
    count = 0
    for round in range(6):
        temp.append([])
        num = 2**(5-round)
        for i in range(num):
            temp[-1].append(int(np.random.random()>prob_mtx[x[2*i],x[2*i+1]]))
        new = x[np.add(temp[-1],range(0,2*num,2))]
        tourns[t,count:count+num] = new
        count += num
        x = new

fig, ax = plt.subplots(2,3)
b = np.arange(16)
ax[0,0].bar(b, np.bincount(seeds[tourns[:,62].astype(int)],minlength=17)[:-1]/T, align='edge', width=.4)
ax[0,0].bar(b+1.4, history[0], align='edge', width=.4)
ax[0,0].set_title('Champion')
ax[0,1].bar(b, np.bincount(seeds[tourns[:,60:62].astype(int)].flatten(),minlength=17)[:-1]/T, align='edge', width=.4)
ax[0,1].bar(b+1.4, history[1], align='edge', width=.4)
ax[0,1].set_title('Championship')
ax[0,2].bar(b, np.bincount(seeds[tourns[:,56:60].astype(int)].flatten(),minlength=17)[:-1]/T, align='edge', width=.4)
ax[0,2].bar(b+1.4, history[2], align='edge', width=.4)
ax[0,2].set_title('Final Four')
ax[1,0].bar(b, np.bincount(seeds[tourns[:,48:56].astype(int)].flatten(),minlength=17)[:-1]/T, align='edge', width=.4)
ax[1,0].bar(b+1.4, history[3], align='edge', width=.4)
ax[1,0].set_title('Elite Eight')
ax[1,1].bar(b, np.bincount(seeds[tourns[:,32:48].astype(int)].flatten(),minlength=17)[:-1]/T, align='edge', width=.4)
ax[1,1].bar(b+1.4, history[4], align='edge', width=.4)
ax[1,1].set_title('Sweet Sixteen')
ax[1,2].bar(b, np.bincount(seeds[tourns[:,:32].astype(int)].flatten(),minlength=17)[:-1]/T, align='edge', width=.4)
ax[1,2].bar(b+1.4, history[5], align='edge', width=.4)
ax[1,2].set_title('Round of 32')
for a in ax.flatten():
    a.set_ylim(0,4)
    a.set_xticks(np.arange(16)+1)

plt.show()