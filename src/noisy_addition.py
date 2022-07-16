# noisy_addition.py         A set of functions to compute the output probabilities of 
#                           noisy half- and full-adders
#
# 2022 written by Ralf Herbrich
# Hasso-Plattner Institute

import matplotlib.pyplot as plt
import numpy as np

def pand(alpha,beta):
    """Computes the distribution of the output of an AND with asymmetric noise"""
    return {
        (0,0): { 0: 1-alpha*alpha,          1: alpha*alpha}, 
        (0,1): { 0: 1-alpha*(1-beta),       1: alpha*(1-beta)}, 
        (1,0): { 0: 1-alpha*(1-beta),       1: alpha*(1-beta)}, 
        (1,1): { 0: 1-(1-beta)*(1-beta),    1: (1-beta)*(1-beta)}
    } 

def por(alpha,beta):
    """Computes the distribution of the output of an OR with asymmetric noise"""
    return {
        (0,0): { 0: (1-alpha)*(1-alpha),    1: 1-(1-alpha)*(1-alpha)}, 
        (0,1): { 0:(1-alpha)*beta,          1: 1-(1-alpha)*beta}, 
        (1,0): { 0:(1-alpha)*beta,          1: 1-(1-alpha)*beta}, 
        (1,1): { 0:beta*beta,               1: 1-beta*beta}
    } 

def pnot(alpha,beta):
    """Computes the distribution of the output of a NOT with asymmetric noise"""
    return {
        0: { 0: alpha, 1: 1-alpha }, 
        1: { 0: 1-beta, 1: beta } 
    } 

def safe_by_or(f_map,or_map):
    """Computes the distribution of the output of a AND/OR function which has been made safe by replication and redundancy resolution via a probabilistic OR"""
    return  {k: { 
                    0:  f_map[k][0]*f_map[k][0]*or_map[(0,0)][0] + 
                        f_map[k][0]*f_map[k][1]*or_map[(0,1)][0] + 
                        f_map[k][1]*f_map[k][0]*or_map[(1,0)][0] + 
                        f_map[k][1]*f_map[k][1]*or_map[(1,1)][0],
                    1:  f_map[k][0]*f_map[k][0]*or_map[(0,0)][1] + 
                        f_map[k][0]*f_map[k][1]*or_map[(0,1)][1] + 
                        f_map[k][1]*f_map[k][0]*or_map[(1,0)][1] + 
                        f_map[k][1]*f_map[k][1]*or_map[(1,1)][1]
               } for k, v in f_map.items()
            }

def recursive_safe(f_map,or_map,n):
    """Applies the save_by_or function recursively to reduce the error rate of the noise AND and ORs"""
    f_out = f_map
    for i in range(n):
        f_out = safe_by_or(f_out,or_map)
    return f_out

def half_adder(and1_map,and2_map,not_map,or_map):
    """Computes the distribution of the output of a half-adder"""
    out = {}
    for a in [0,1]:
        for b in [0,1]:
            k = (a,b)
            v = {}
            for s in [0,1]:
                for c_out in [0,1]:
                    kv = (s,c_out)
                    p = 0
                    for d in [0,1]:
                        for e in [0,1]:
                            p = p + and1_map[(a,b)][c_out]*or_map[(a,b)][d]*not_map[c_out][e]*and2_map[(d,e)][s]
                    v[kv] = p
            out[k] = v
    return out

def full_adder(ha1_map,ha2_map,or_map):
    """Computes the distribution of the output of a full-adder"""
    out = {}
    for a in [0,1]:
        for b in [0,1]:
            for c_in in [0,1]:
                k = (a,b,c_in)
                v = {}
                for s in [0,1]:
                    for c_out in [0,1]:
                      kv = (s,c_out)
                      p = 0
                      for d in [0,1]:
                        for e in [0,1]:
                            for f in [0,1]:
                                p = p + ha1_map[(a,b)][(d,e)]*ha2_map[(d,c_in)][(s,f)]*or_map[(e,f)][c_out] 
                      v[kv] = p
                out[k] = v
    return out

def four_bit_adder(ha_map,fa1_map,fa2_map,fa3_map):
    """Computes the distribution of the output of a 4-bit adder"""
    out = {}
    for a0 in [0,1]:
        for a1 in [0,1]:
            for a2 in [0,1]:
                for a3 in [0,1]:
                    for b0 in [0,1]:
                        for b1 in [0,1]:
                            for b2 in [0,1]:
                                for b3 in [0,1]:
                                    k = (a0,a1,a2,a3,b0,b1,b2,b3)
                                    v = {}
                                    for s0 in [0,1]:
                                        for s1 in [0,1]:
                                            for s2 in [0,1]:
                                                for s3 in [0,1]:
                                                    for c3_out in [0,1]:
                                                        kv = (s0,s1,s2,s3,c3_out)
                                                        p = 0
                                                        for c0_out in [0,1]:
                                                            for c1_out in [0,1]:
                                                                for c2_out in [0,1]:
                                                                    p = p + \
                                                                        ha_map[(a0,b0)][(s0,c0_out)] * \
                                                                        fa1_map[(a1,b1,c0_out)][(s1,c1_out)] * \
                                                                        fa2_map[(a2,b2,c1_out)][(s2,c2_out)] * \
                                                                        fa3_map[(a3,b3,c2_out)][(s3,c3_out)]
                                                        v[kv] = p
                                    out[k] = v
    return(out)

def plot_basic_logic(d=8):
    """Plots the probability distributions of basic logic functions"""
    beta = np.linspace(0, 0.5, 100)
    plt.plot(beta, [pand(0,b)[(0,0)][1] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [pand(0,b)[(0,1)][1] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [pand(0,b)[(1,0)][1] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [pand(0,b)[(1,1)][1] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='black')

    plt.plot(beta, [recursive_safe(pand(0,b),por(0,b),d)[(0,0)][1] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [recursive_safe(pand(0,b),por(0,b),d)[(0,1)][1] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [recursive_safe(pand(0,b),por(0,b),d)[(1,0)][1] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [recursive_safe(pand(0,b),por(0,b),d)[(1,1)][1] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='black')

    plt.legend(['P(and=1|a=0,b=0)', 'P(and=1|a=0,b=1)', 'P(and=1|a=1,b=0)', 'P(and=1|a=1,b=1)'])
    plt.xlabel(r'$\beta$')
    plt.ylabel('P(and=1|a,b)')
    plt.title('Probability Distribution of AND with k-redundancy')

    plt.grid()
    plt.show()

def plot_half_adder(inp=(1,1),d=8):
    """Plots the probability distributions of basic logic functions"""
    beta = np.linspace(0, 0.5, 100)

    plt.plot(beta, [half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b))[inp][(0,0)] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b))[inp][(1,0)] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b))[inp][(0,1)] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b))[inp][(1,1)] for b in beta] , linestyle='-', linewidth=2.5, marker='', color='black')

    plt.plot(beta, [half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),recursive_safe(por(0,b),por(0,b),d))[inp][(0,0)] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='red')
    plt.plot(beta, [half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),recursive_safe(por(0,b),por(0,b),d))[inp][(1,0)] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='blue')
    plt.plot(beta, [half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),recursive_safe(por(0,b),por(0,b),d))[inp][(0,1)] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='green')
    plt.plot(beta, [half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),recursive_safe(por(0,b),por(0,b),d))[inp][(1,1)] for b in beta] , linestyle='--', linewidth=2.5, marker='', color='black')

    plt.legend([r'P(s=0,$c_{out}$=0|a,b)', r'P(s=1,$c_{out}$=0|a,b)', r'P(s=0,$c_{out}$=1|a,b)', r'P(s=1,$c_{out}$=1|a,b)'])
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'P(s,$c_{out}$|a,b)')
    plt.title('Probability Distribution of half-adder with k-redundancy')

    plt.grid()
    plt.show()

def plot_full_adder(inp=(1,1,0),d=8):
    """Plots the probability distributions of basic logic functions"""
    beta = np.linspace(0, 0.5, 100)

    plt.plot(beta, [
        full_adder(half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                por(0,b))[inp][(0,0)]
        for b in beta] , linestyle='-', marker='', color='red', linewidth=2.5)
    plt.plot(beta, [
        full_adder(half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                por(0,b))[inp][(0,1)]
        for b in beta] , linestyle='-', marker='', color='blue', linewidth=2.5)
    plt.plot(beta, [
        full_adder(half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                por(0,b))[inp][(1,0)]
        for b in beta] , linestyle='-', marker='', color='green', linewidth=2.5)
    plt.plot(beta, [
        full_adder(half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                half_adder(pand(0,b),pand(0,b),pnot(0,b),por(0,b)),
                por(0,b))[inp][(1,1)]
        for b in beta] , linestyle='-', marker='', color='black', linewidth=2.5)

    plt.plot(beta, [
        full_adder(half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                por(0,b))[inp][(0,0)]
        for b in beta] , linestyle='--', marker='', color='red', linewidth=2.5)
    plt.plot(beta, [
        full_adder(half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                por(0,b))[inp][(0,1)]
        for b in beta] , linestyle='--', marker='', color='blue', linewidth=2.5)
    plt.plot(beta, [
        full_adder(half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                por(0,b))[inp][(1,0)]
        for b in beta] , linestyle='--', marker='', color='green', linewidth=2.5)
    plt.plot(beta, [
        full_adder(half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                half_adder(recursive_safe(pand(0,b),por(0,b),d),recursive_safe(pand(0,b),por(0,b),d),pnot(0,b),por(0,b)),
                por(0,b))[inp][(1,1)]
        for b in beta] , linestyle='--', marker='', color='black', linewidth=2.5)

    plt.legend([r'P(s=0,$c_{out}$=0|a,b,$c_{in}$)', r'P(s=1,$c_{out}$=0|a,b,$c_{in}$)', r'P(s=0,$c_{out}$=1|a,b,$c_{in}$)', r'P(s=1,$c_{out}$=1|a,b,$c_{in}$)'])
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'P(s,$c_{out}$|a,b,$c_{in}$)')
    plt.title('Probability Distribution of a full-adder with k-redundancy')

    plt.grid()
    plt.show()

def plot_4bit_adder(alpha = 0.05, beta = 0.05, a = 3, b = 7):
    ha_map = half_adder(pand(alpha,beta),pand(alpha,beta),pnot(alpha,beta),por(alpha,beta))
    fa_map = full_adder(ha_map, ha_map, por(alpha,beta))
    fb_map = four_bit_adder(ha_map,fa_map,fa_map,fa_map)

    key = (a & 1, (a>>1) & 1, (a>>2) & 1, (a>>3) & 1,b & 1, (b>>1) & 1, (b>>2) & 1, (b>>3) & 1)
    hist = dict(sorted({ c4*16+c3*8+c2*4+c1*2+c0: v for ((c0,c1,c2,c3,c4),v) in fb_map[key].items() }.items()))

    plt.bar(hist.keys(), hist.values())
    plt.xlabel(r'sum')
    plt.ylabel('P(sum|a={0},b={1})'.format(a,b))
    plt.title(r'Probability Distribution over sum of {0} and {1} ($\alpha$={2},$\beta$={3})'.format(a,b,alpha,beta))
    plt.show()


def print_distribution(p_map = full_adder(half_adder(pand(0,0),pand(0,0),pnot(0,0),por(0,0)),
                                          half_adder(pand(0,0),pand(0,0),pnot(0,0),por(0,0)),
                                          por(0,0))):
    """Outputs a probabilistic logic function on screen"""
    for (k,v) in p_map.items():
        print(k, v)

def check_distribution(p_map = full_adder(half_adder(pand(0,0),pand(0,0),pnot(0,0),por(0,0)),
                                          half_adder(pand(0,0),pand(0,0),pnot(0,0),por(0,0)),
                                          por(0,0))):
    for (k,v) in p_map.items():
        if(abs(sum([v2 for (k2,v2) in v.items()])-1.0) > 1e-4):
            print("normalization error for key", k, ": ", sum([v2 for (k2,v2) in v.items()]))
            exit()
    print("ok")

# plot_basic_logic()
# plot_half_adder()
# plot_full_adder()
plot_4bit_adder()

# print_distribution(fb_map)
# check_distribution(fb_map)


