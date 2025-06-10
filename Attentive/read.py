import numpy as np
import matplotlib.pyplot as plt
# with open('./pareto_origin.txt', 'r') as f:
#     ls = f.readlines()
# baseline = []
# for l in ls:
#     if '0' in l:
#         _, d = eval(l)
#         baseline.append([d['acc1'], d['flops']])
# # res = np.loadtxt('./flops400_465.txt', delimiter=',')
# baseline = np.array(baseline)
# res = np.loadtxt('./pareto_4404.txt', delimiter=',')[:,:2]

# mask = (res[:,0]<1020) & (res[:,0]>1010)
# plt.clf()
# # plt.ylim([74.5,81.5])

# plt.plot(baseline[:,1],baseline[:,0], '-o', color='k')
# plt.scatter(res[:,0][mask],res[:,1][mask], color='r')
# plt.savefig('./out.png')
# print()
# res[2699]


with open('./pareto_origin.txt', 'r') as f:
    ls = f.readlines()
baseline = []
for l in ls:
    if '0' in l:
        _, d = eval(l)
        baseline.append([d['acc1'], d['flops']])
# res = np.loadtxt('./flops400_465.txt', delimiter=',')
baseline = np.array(baseline)
res = np.loadtxt('./flops1020-872.txt', delimiter=',')[:,:2]
# res = np.loadtxt('./pareto_4404.txt', delimiter=',')[:,:2]


plt.clf()
# plt.ylim([74.5,81.5])

plt.plot(baseline[:,1],baseline[:,0], '-o', color='k')
plt.scatter(res[:,0],res[:,1], color='r')
plt.savefig('./out.png')
print()