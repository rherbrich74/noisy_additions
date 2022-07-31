# noisy_addition_mitigation.py         A set of functions to compute the output probabilities of 
#                                      mitigated noisy half-, full- and 4-bit-adders
#
# 2022 written by Ralf Herbrich
# Hasso-Plattner Institute

import matplotlib.pyplot as plt
import numpy as np

def pnand(alpha,beta):
    """Computes the distribution of the output of an AND with asymmetric noise"""
    return {
        (0,0): { 0: alpha*alpha,          1: 1-alpha*alpha}, 
        (0,1): { 0: alpha*(1-beta),       1: 1-alpha*(1-beta)}, 
        (1,0): { 0: alpha*(1-beta),       1: 1-alpha*(1-beta)}, 
        (1,1): { 0: (1-beta)*(1-beta),    1: 1-(1-beta)*(1-beta)}
    } 

def pnor(alpha,beta):
    """Computes the distribution of the output of an NOR with asymmetric noise"""
    return {
        (0,0): { 0:1-(1-alpha)*(1-alpha),   1: (1-alpha)*(1-alpha)}, 
        (0,1): { 0:1-(1-alpha)*beta,        1: (1-alpha)*beta}, 
        (1,0): { 0:1-(1-alpha)*beta,        1: (1-alpha)*beta}, 
        (1,1): { 0:1-beta*beta,             1: beta*beta}
    } 

def pnot(alpha,beta):
    """Computes the distribution of the output of a NOT with asymmetric noise"""
    return {
        0: { 0: alpha, 1: 1-alpha }, 
        1: { 0: 1-beta, 1: beta } 
    } 

def invert(f_map,not_map):
    """Computes the distribution of the output of a single-valued logic gate is followed by a 
    probabilistic NOT gate"""
    return  {k: { 
                    0: v[0]*not_map[(0)][0] + v[1]*not_map[(1)][0],
                    1: v[0]*not_map[(0)][1] + v[1]*not_map[(1)][1]
               } for k, v in f_map.items()
            }

def plot_basic_logic(alpha=0):
    """Plots the probability distributions of basic logic functions"""
    beta = np.linspace(0, 0.5, 100)
    plt.plot(beta, [pnand(alpha,b)[(0,0)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [pnand(alpha,b)[(0,1)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [pnand(alpha,b)[(1,0)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [pnand(alpha,b)[(1,1)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='black')

    plt.legend(['P(nand=1|a=0,b=0)', 'P(nand=1|a=0,b=1)', 'P(nand=1|a=1,b=0)', 'P(nand=1|a=1,b=1)'])
    plt.xlabel(r'$\beta$')
    plt.ylabel('P(nand=1|a,b)')
    plt.title('Probability Distribution of NAND')

    plt.grid()

def print_distribution(p_map):
    """Outputs a probabilistic logic function on screen"""
    for (k,v) in p_map.items():
        print(k, v)

def check_distribution(p_map):
    for (k,v) in p_map.items():
        if(abs(sum([v2 for (k2,v2) in v.items()])-1.0) > 1e-4):
            print("normalization error for key", k, ": ", sum([v2 for (k2,v2) in v.items()]))
            exit()
    print("ok")

################ TEST CODE #####################

def pnot_safe2(alpha,beta):
    """Computes the distribution of the output of a NOT with asymmetric noise"""
    not_map = pnot(alpha,beta)
    nand_map = pnor(alpha,beta)
    out = {}
    for a in [0,1]:
        k = a
        v = {}
        for n_out in [0,1]:
            kv = n_out
            p = 0
            for n1 in [0,1]:
                for n2 in [0,1]:
                    for d in [0,1]:
                        p = p + not_map[a][n1] * not_map[a][n2] * nand_map[(n1,n2)][d] * not_map[d][n_out]
            v[kv] = p
        out[k] = v
    return out

def pnot_safe3(alpha,beta):
    """Computes the distribution of the output of a NOT with asymmetric noise"""
    not_map = pnot(alpha,beta)
    nand_map = pnor(alpha,beta)
    nor_map = pnor(alpha,beta)
    out = {}
    for a in [0,1]:
        k = a
        v = {}
        for n_out in [0,1]:
            kv = n_out
            p = 0
            for n1 in [0,1]:
                for n2 in [0,1]:
                    for n3 in [0,1]:
                        for d in [0,1]:
                            for e in [0,1]:
                                p = p + not_map[a][n1] * not_map[a][n2] * not_map[a][n3] * \
                                    not_map[n1][d] * nand_map[(n2,n3)][e] * nor_map[(d,e)][n_out]
            v[kv] = p
        out[k] = v
    return out

def pnot_safe4(alpha,beta):
    """Computes the distribution of the output of a NOT with asymmetric noise"""
    not_map = pnot(alpha,beta)
    nand_map = pnor(alpha,beta)
    nor_map = pnor(alpha,beta)
    out = {}
    for a in [0,1]:
        k = a
        v = {}
        for n_out in [0,1]:
            kv = n_out
            p = 0
            for n1 in [0,1]:
                for n2 in [0,1]:
                    for n3 in [0,1]:
                        for n4 in [0,1]:
                            for d in [0,1]:
                                for e in [0,1]:
                                    p = p + not_map[a][n1] * not_map[a][n2] * not_map[a][n3] * not_map[a][n4] * \
                                        nand_map[(n1,n2)][d] * nand_map[(n3,n4)][e] * nor_map[(d,e)][n_out]
            v[kv] = p
        out[k] = v
    return out

def pnot_safe(alpha,beta,not_map):
    """Computes the distribution of the output of a NOT with asymmetric noise"""
    nand_map = pnor(alpha,beta)
    nor_map = pnor(alpha,beta)
    out = {}
    for a in [0,1]:
        k = a
        v = {}
        for n_out in [0,1]:
            kv = n_out
            p = 0
            for n1 in [0,1]:
                for n2 in [0,1]:
                    for n3 in [0,1]:
                        for n4 in [0,1]:
                            for d in [0,1]:
                                for e in [0,1]:
                                    p = p + not_map[a][n1] * not_map[a][n2] * not_map[a][n3] * not_map[a][n4] * \
                                        nand_map[(n1,n2)][d] * nand_map[(n3,n4)][e] * nor_map[(d,e)][n_out]
            v[kv] = p
        out[k] = v
    return out

alpha = 0
beta = 0.1
not_map = pnot(alpha,beta)
not_map2 = pnot_safe2(alpha,beta)
not_map3 = pnot_safe3(alpha,beta)
not_map4 = pnot_safe4(alpha,beta)
not_map16 = pnot_safe(alpha,beta,not_map4)
not_map64 = pnot_safe(alpha,beta,not_map16)
check_distribution(not_map)
check_distribution(not_map2)
check_distribution(not_map3)
check_distribution(not_map4)
check_distribution(not_map16)
check_distribution(not_map64)
print('N=1')
print_distribution(not_map)
print('N=2')
print_distribution(not_map2)
print('N=3')
print_distribution(not_map3)
print('N=4')
print_distribution(not_map4)
print('N=16')
print_distribution(not_map16)
print('N=64')
print_distribution(not_map64)

beta = np.linspace(0, 0.5, 100)
perfect_out = {k:max(v,key=v.get) for (k,v) in pnot(0,0).items()}
error = [np.average([1-v[perfect_out[k]] for (k,v) in pnot(0,b).items()]) for b in beta]
error2 = [np.average([1-v[perfect_out[k]] for (k,v) in pnot_safe2(0,b).items()]) for b in beta]
error3 = [np.average([1-v[perfect_out[k]] for (k,v) in pnot_safe3(0,b).items()]) for b in beta]
error4 = [np.average([1-v[perfect_out[k]] for (k,v) in pnot_safe4(0,b).items()]) for b in beta]
error16 = [np.average([1-v[perfect_out[k]] for (k,v) in pnot_safe(0,b,pnot_safe4(0,b)).items()]) for b in beta]
error64 = [np.average([1-v[perfect_out[k]] for (k,v) in pnot_safe(0,b,pnot_safe(0,b,pnot_safe4(0,b))).items()]) for b in beta]
plt.plot(beta, error, linestyle='-', linewidth=2.5, marker='', color='red')
plt.plot(beta, error2, linestyle='--', linewidth=2.5, marker='', color='red')
plt.plot(beta, error3, linestyle='-.', linewidth=2.5, marker='', color='red')
plt.plot(beta, error4, linestyle=':', linewidth=2.5, marker='', color='red')
plt.plot(beta, error16, linestyle=':', linewidth=2.5, marker='', color='green')
plt.plot(beta, error64, linestyle=':', linewidth=2.5, marker='', color='blue')
plt.legend(['without replication (N=1)',
            'with replication (N=2)',
            'with replication (N=3)',
            'with replication (N=4)',
            'with replication (N=16)',
            'with replication (N=64)'])
plt.xlabel(r'$\beta$')
plt.ylabel(r'P(error|$\beta$)')
plt.title('Error Probability of NOT')

plt.grid()
plt.show()