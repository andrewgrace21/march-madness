import numpy as np
from scipy.stats import norm
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt

T = 5000
N = 50
G = 100
C = 30

teams = np.array(['Duke', 'Siena', 'Ohio St.', 'TCU', 'St. Johns', 'Northern Iowa', 'Kansas', 'Cal Baptist', 'Louisville', 'South Florida', 'Michigan St.', 'North Dakota St.', 'UCLA', 'UCF', 'UConn', 'Furman', 'Florida', 'LEHIGH/PVAMU', 'Clemson', 'Iowa', 'Vanderbilt', 'McNeese', 'Nebraska', 'Troy', 'North Carolina', 'VCU', 'Illinois', 'Penn', 'Saint Marys', 'Texas A&M', 'Houston', 'Idaho', 'Arizona', 'Long Island', 'Villanova', 'Utah St.', 'Wisconsin', 'High Point', 'Arkansas', 'Hawaii', 'BYU', 'NC ST/TEXAS', 'Gonzaga', 'Kennesaw St.', 'Miami', 'Missouri', 'Purdue', 'Queens', 'Michigan', 'HOW/UMBC', 'Georgia', 'Saint Louis', 'Texas Tech', 'Akron', 'Alabama', 'Hofstra', 'Tennessee', 'SMU/MIA OH', 'Virginia', 'Wright St.', 'Kentucky', 'Santa Clara', 'Iowa St.', 'Tennessee St.'])
NetRtg = np.array([14.29, -9.48, 13.67, 11.04, 11.52, 1.27, 16.95, -1.95, 12.55, 3.04, 13.69, -5.83, 12.22, 11.91, 12.01, -6.29, 16.01, -9.1, 10.53, 11.37, 14.56, -1.86, 11.59, -3.23, 11.46, 3.49, 13.64, -.69, 4.99, 11.15, 13.58, -1.67, 14.97, -9.97, 10.37, 7.13, 13.93, -9.23, 14.95, -3.37, 14.27, 12.84, 5.89, -2.23, 7.99, 11.51, 15.88, -5.66, 16.65, -14.74, 10.78, 1.03, 15.64, -3.65, 16.75, -.91, 14.77, 9.77, 9.95, -4.03, 15.92, 6.02, 12.44, -8.26])
df = pd.read_csv("Yahoo Pick Distribution 2026.csv")
ppl_pick = np.array(df[['round1','round2','round3','round4','round5','round6']])
seeds = np.array([1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]*4)
full_seed = np.array([3,62,28,40,15,50,19,52,22,43,8,54,27,44,10,60,5,67.5,32,31,14,47,12,56,29,42,7,59,23,36,4,58,1,66,25,30,18,49,13,53,24,41.5,11,57,34,41,9,63,2,62.5,33,35,20,48,16,51,21,41.5,17,55,26,39,6,21])
history = np.array([[0.65, 0.13, 0.10, 0.05, 0.00, 0.03, 0.03, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], [1.03, 0.33, 0.28, 0.10, 0.10, 0.05, 0.03, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], [1.65, 0.80, 0.42, 0.38, 0.22, 0.08, 0.08, 0.15, 0.05, 0.02, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00], [2.68, 1.80, 1.02, 0.62, 0.30, 0.42, 0.25, 0.22, 0.12, 0.22, 0.25, 0.05, 0.00, 0.00, 0.02, 0.00], [3.40, 2.58, 2.10, 1.92, 1.38, 1.18, 0.72, 0.40, 0.20, 0.60, 0.68, 0.55, 0.15, 0.05, 0.10, 0.00], [3.95, 3.72, 3.42, 3.18, 2.58, 2.45, 2.45, 1.92, 2.08, 1.55, 1.55, 1.42, 0.82, 0.58, 0.28, 0.05]])
points = np.array(32*[10] + 16*[20] + 8*[40] + 4*[80] + 2*[160] + 1*[320])
prize = np.array([200,300,500]) #3,2,1

def win_prob(adjEM_A, adjEM_B, seed_A, seed_B, poss=67, sigma=11):
    mu = (adjEM_A - adjEM_B) * poss / 100 + (seed_B**.0625 - seed_A**.0625) * 48
    return norm.cdf(mu / sigma)

def find_bracket(bin):
    out = []
    x = np.arange(64)
    for l in range(6):
        x = x[np.add(bin[l], range(0,2**(6-l),2))]
        out.extend(x)
    return np.array(out)

best = [32*[0], 4*[0,1,1,1], 4*[0,1], 4*[0], 2*[0], [0]]

prob_mtx = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        prob_mtx[i][j] = win_prob(NetRtg[i],NetRtg[j],seeds[i],seeds[j])

tourns = np.zeros((T,63))
ops = np.zeros((T,N,63))
top3 = np.zeros((T,3))
all_score = np.zeros((T,N))
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
    
    for op in range(N):
        temp = []
        x = np.arange(64)
        count = 0
        for round in range(6):
            temp.append([])
            num = 2**(5-round)
            for i in range(num):
                prob = ppl_pick[x[2*i],round] / (ppl_pick[x[2*i],round]+ppl_pick[x[2*i+1],round])
                temp[-1].append(int(np.random.random()>prob))
            new = x[np.add(temp[-1],range(0,2*num,2))]
            ops[t,op,count:count+num] = new
            count += num
            x = new

    scores = sorted([sum((tourns[t]==ops[t,n]).astype(int)*points) for n in range(N)])
    top3[t] = scores[-3:]

for gen in range(G):
    children = [best]
    n_mut = poisson.rvs(mu=1, size=C)
    for c in range(C):
        temp = [arr.copy() for arr in best]
        for m in range(n_mut[c]+1):
            l = np.random.randint(6)
            s = np.random.randint(2**(5-l))
            temp[l][s] = (temp[l][s]+1)%2
        children.append(temp)
    actual = np.array([find_bracket(c) for c in children])
    pools = np.random.choice(range(T),int(T/1))
    pfm = np.array([[sum((tourns[p]==actual[c]).astype(int)*points) for c in range(C+1)] for p in pools])
    check = np.array([(tourns[p]==actual[0]) for p in pools])
    value = np.mean(np.max((pfm[:,:,None]>top3[pools,None,:]).astype(int) * prize[None,None,:], axis=2), axis=0)
    best = children[np.argmax(value)].copy()
    #print(find_bracket(best))

opt = teams[find_bracket(best)]
print(f"\n\nOptimal Bracket\n{'-'*16}\nRound of 32: {', '.join(opt[:32])}\nSweet Sixteen: {', '.join(opt[32:48])}\nElite Eight: {', '.join(opt[48:56])}\nFinal Four: {', '.join(opt[56:60])}\nChampionship: {', '.join(opt[60:62])}\nChampion: {opt[62]}\n\nScore -> {np.mean(pfm[:,np.argmax(value)])}")

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