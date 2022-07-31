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

def psafe(alpha,beta,f_map):
    """Takes a single-output circuit and quadruples it to add enough redundancy"""
    nor_map = pnor(alpha,beta)
    out = {}
    for k,_ in f_map.items():
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
                                    p = p + f_map[k][n1] * f_map[k][n2] * f_map[k][n3] * f_map[k][n4] * \
                                        nor_map[(n1,n2)][d] * nor_map[(n3,n4)][e] * nor_map[(d,e)][n_out]
            v[kv] = p
        out[k] = v
    return out

def plot_basic_logic(beta_max=0.5):
    """Plots the probability distributions of basic logic functions"""
    beta = np.linspace(0, beta_max, 100)
    plt.plot(beta, [pnand(0,b)[(0,0)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [pnand(0,b)[(0,1)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [pnand(0,b)[(1,0)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [pnand(0,b)[(1,1)][1] for b in beta], linestyle='-', linewidth=2.5, marker='', color='black')

    plt.plot(beta, [psafe(0,b,pnand(0,b))[(0,0)][1] for b in beta], linestyle='--', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [psafe(0,b,pnand(0,b))[(0,1)][1] for b in beta], linestyle='--', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [psafe(0,b,pnand(0,b))[(1,0)][1] for b in beta], linestyle='--', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [psafe(0,b,pnand(0,b))[(1,1)][1] for b in beta], linestyle='--', linewidth=2.5, marker='', color='black')

    plt.legend(['P(nand=1|a=0,b=0)', 'P(nand=1|a=0,b=1)', 'P(nand=1|a=1,b=0)', 'P(nand=1|a=1,b=1)',
                r'P(nand$_2$=1|a=0,b=0)', r'P(nand$_2$=1|a=0,b=1)', r'P(nand$_2$=1|a=1,b=0)', r'P(nand$_2$=1|a=1,b=1)'])
    plt.xlabel(r'$\beta$')
    plt.ylabel('P(nand=1|a,b)')
    plt.title('Probability Distribution of NAND')

    plt.grid()

def plot_error(beta_max = 0.5,gate="NOT"):
    beta = np.linspace(0, beta_max, 100)
    if (gate == "NOT"):
        perfect_out = {k:max(v,key=v.get) for (k,v) in pnot(0,0).items()}
        error = [np.average([1-v[perfect_out[k]] for (k,v) in pnot(0,b).items()]) for b in beta]
        error4 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,pnot(0,b)).items()]) for b in beta]
        error16 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,psafe(0,b,pnot(0,b))).items()]) for b in beta]
        error64 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,psafe(0,b,psafe(0,b,pnot(0,b)))).items()]) for b in beta]
    if (gate == "NOR"):
        perfect_out = {k:max(v,key=v.get) for (k,v) in pnor(0,0).items()}
        error = [np.average([1-v[perfect_out[k]] for (k,v) in pnor(0,b).items()]) for b in beta]
        error4 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,pnor(0,b)).items()]) for b in beta]
        error16 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,psafe(0,b,pnor(0,b))).items()]) for b in beta]
        error64 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,psafe(0,b,psafe(0,b,pnor(0,b)))).items()]) for b in beta]
    if (gate == "NAND"):
        perfect_out = {k:max(v,key=v.get) for (k,v) in pnand(0,0).items()}
        error = [np.average([1-v[perfect_out[k]] for (k,v) in pnand(0,b).items()]) for b in beta]
        error4 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,pnand(0,b)).items()]) for b in beta]
        error16 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,psafe(0,b,pnand(0,b))).items()]) for b in beta]
        error64 = [np.average([1-v[perfect_out[k]] for (k,v) in psafe(0,b,psafe(0,b,psafe(0,b,pnand(0,b)))).items()]) for b in beta]
    plt.plot(beta, error, linestyle='-', linewidth=2.5, marker='', color='red')
    plt.plot(beta, error4, linestyle='--', linewidth=2.5, marker='', color='red')
    plt.plot(beta, error16, linestyle='-.', linewidth=2.5, marker='', color='red')
    plt.plot(beta, error64, linestyle=':', linewidth=2.5, marker='', color='red')
    plt.legend(['without replication (N=1)',
                'with replication (N=4)',
                'with replication (N=16)',
                'with replication (N=64)'])
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'P(error|$\beta$)')
    if (gate == "NOT"):
        plt.title('Error Probability of NOT')
    if (gate == "NOR"):
        plt.title('Error Probability of NOR')
    if (gate == "NAND"):
        plt.title('Error Probability of NAND')

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

plot_basic_logic(beta_max = 0.5)
plt.show()

plot_error(beta_max = 0.15, gate="NOT")
plt.show()
plot_error(beta_max = 0.15, gate="NOR")
plt.show()
plot_error(beta_max = 0.15, gate="NAND")
plt.show()
