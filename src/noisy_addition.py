# noisy_addition.py         A set of functions to compute the output probabilities of 
#                           noisy half-, full- and 4-bit-adders
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

def half_adder(nand_map,nor_map,not_map):
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
                            for f in [0,1]:
                                for g in [0,1]:
                                    p = p + nand_map[(a,b)][e]*nor_map[(a,b)][d]*not_map[e][c_out]*not_map[d][f]*nand_map[(e,f)][g]*not_map[g][s]
                    v[kv] = p
            out[k] = v
    return out

def full_adder(ha_map,nor_map,not_map):
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
                                for g in [0,1]:
                                    p = p + ha_map[(a,b)][(d,e)]*ha_map[(d,c_in)][(s,f)]*nor_map[(e,f)][g]*not_map[g][c_out] 
                      v[kv] = p
                out[k] = v
    return out

def four_bit_adder(ha_map,fa_map):
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
                                                                        fa_map[(a1,b1,c0_out)][(s1,c1_out)] * \
                                                                        fa_map[(a2,b2,c1_out)][(s2,c2_out)] * \
                                                                        fa_map[(a3,b3,c2_out)][(s3,c3_out)]
                                                        v[kv] = p
                                    out[k] = v
    return(out)

def plot_basic_logic(alpha=0,d=4):
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

def plot_half_adder_error(beta_max=0.5):
    """Plots the probability distributions of errors for the half-adder"""
    beta = np.linspace(0, beta_max, 100)
    perfect_out = {k:max(v,key=v.get) for (k,v) in half_adder(pnand(0,0),pnor(0,0),pnot(0,0)).items()}
    error = [np.average([1-v[perfect_out[k]] for (k,v) in half_adder(pnand(0,b),pnor(0,b),pnot(0,b)).items()]) \
             for b in beta]
    plt.plot(beta, error, linestyle='-', linewidth=2.5, marker='', color='red')
    plt.xlabel(r'$\beta$')
    plt.ylabel([r'P(error|$beta$)'])
    plt.title('Error Probability of a half-adder')

    plt.grid()

def plot_full_adder(inp=(1,0,1)):
    """Plots the probability distributions of a full adder"""
    beta = np.linspace(0, 0.5, 100)

    P = [[full_adder(half_adder(pnand(0,b),pnor(0,b),pnot(0,b)),pnor(0,b),pnot(0,b))[inp][(0,0)] for b in beta]]
    P += [[full_adder(half_adder(pnand(0,b),pnor(0,b),pnot(0,b)),pnor(0,b),pnot(0,b))[inp][(1,0)] for b in beta]]
    P += [[full_adder(half_adder(pnand(0,b),pnor(0,b),pnot(0,b)),pnor(0,b),pnot(0,b))[inp][(0,1)] for b in beta]]
    P += [[full_adder(half_adder(pnand(0,b),pnor(0,b),pnot(0,b)),pnor(0,b),pnot(0,b))[inp][(1,1)] for b in beta]]
    P_stack = np.cumsum(P, axis=0)

    plt.fill_between(beta, 0, P_stack[0,:], facecolor="red")
    plt.fill_between(beta, P_stack[0,:], P_stack[1,:], facecolor="blue")
    plt.fill_between(beta, P_stack[1,:], P_stack[2,:], facecolor="green")
    plt.fill_between(beta, P_stack[2,:], P_stack[3,:], facecolor="black")
    plt.legend([r'P(s=0,$c_{out}$=0)',r'P(s=1,$c_{out}$=0)',r'P(s=0,$c_{out}$=1)',r'P(s=1,$c_{out}$=1)'], loc='lower right')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'P(s,$c_{out}$|a,b,$c_{in}$|a=%d,b=%d,$c_{in}$=%d)' % (inp[0],inp[1],inp[2]))
    plt.title('Probability Distribution of a full-adder')

    plt.grid()

def plot_4bit_adder(alpha = 0, beta = 0.15, a = 14, b = 7):
    """Plots the distribution of sums of the 4bit adder for two specific inputs"""
    ha_map = half_adder(pnand(alpha,beta),pnor(alpha,beta),pnot(alpha,beta))
    fa_map = full_adder(ha_map,pnor(alpha,beta),pnot(alpha,beta))
    fb_map = four_bit_adder(ha_map,fa_map)

    key = (a & 1, (a>>1) & 1, (a>>2) & 1, (a>>3) & 1,b & 1, (b>>1) & 1, (b>>2) & 1, (b>>3) & 1)
    hist = dict(sorted({ c4*16+c3*8+c2*4+c1*2+c0: v for ((c0,c1,c2,c3,c4),v) in fb_map[key].items() }.items()))

    plt.bar(hist.keys(), hist.values())
    plt.xlabel(r'sum')
    plt.ylabel('P(sum|a={0},b={1})'.format(a,b))
    plt.title(r'Probability Distribution over sum of {0} and {1} ($\alpha$={2},$\beta$={3})'.format(a,b,alpha,beta))

def plot_4bit_adder_dist(alpha1 = 0, beta1 = 0.05, alpha2 = 0, beta2 = 0.1):
    """Plots the whole distribution of the 4bit adders"""
    def compute_map(fb_map):
        A = np.zeros((32,256))
        for ((a0,a1,a2,a3,b0,b1,b2,b3),v) in fb_map.items():
            key1 = a0*1 + a1*2 + a2*4 + a3*8 + b0*16 + b1*32 + b2*64 + b3*128
            for ((c0,c1,c2,c3,c4),p) in v.items():
                key2 = c4*16+c3*8+c2*4+c1*2+c0
                A[key2][key1] = p
        return A

    ha_map0 = half_adder(pnand(0,0),pnor(0,0),pnot(0,0))
    fa_map0 = full_adder(ha_map0,pnor(0,0),pnot(0,0))
    fb_map0 = four_bit_adder(ha_map0,fa_map0)
    A0 = compute_map(fb_map0)

    ha_map1 = half_adder(pnand(alpha1,beta1),pnor(alpha1,beta1),pnot(alpha1,beta1))
    fa_map1 = full_adder(ha_map1,pnor(alpha1,beta1),pnot(alpha1,beta1))
    fb_map1 = four_bit_adder(ha_map1,fa_map1)
    A1 = compute_map(fb_map1)

    ha_map2 = half_adder(pnand(alpha2,beta2),pnor(alpha2,beta2),pnot(alpha2,beta2))
    fa_map2 = full_adder(ha_map2,pnor(alpha2,beta2),pnot(alpha2,beta2))
    fb_map2 = four_bit_adder(ha_map2,fa_map2)
    A2 = compute_map(fb_map2)

    fig, axs = plt.subplots(3)
    # fig.suptitle('4-bit Adder Output Distribution for all 256 inputs to {0,..,31}')
    axs[0].matshow(A0)
    axs[0].title.set_text('Zero Noise')
    axs[1].matshow(A1)
    axs[1].title.set_text(r'$\alpha$={0}, $\beta$={1}'.format(alpha1,beta1))
    axs[2].matshow(A2)
    axs[2].title.set_text(r'$\alpha$={0}, $\beta$={1}'.format(alpha2,beta2))

def print_distribution(p_map = full_adder(half_adder(pnand(0,0),pnor(0,0),pnot(0,0)),pnor(0,0),pnot(0,0))):
    """Outputs a probabilistic logic function on screen"""
    for (k,v) in p_map.items():
        print(k, v)

def check_distribution(p_map = full_adder(half_adder(pnand(0,0),pnor(0,0),pnot(0,0)),pnor(0,0),pnot(0,0))):
    for (k,v) in p_map.items():
        if(abs(sum([v2 for (k2,v2) in v.items()])-1.0) > 1e-4):
            print("normalization error for key", k, ": ", sum([v2 for (k2,v2) in v.items()]))
            exit()
    print("ok")

def gen_paper_plots():
    plot_half_adder(inp=(0,0))
    plt.savefig('media/noisy_half_adder_value_dist_00.eps', format='eps')
    plot_half_adder(inp=(0,1))
    plt.savefig('media/noisy_half_adder_value_dist_01.eps', format='eps')
    plot_half_adder(inp=(1,1))
    plt.savefig('media/noisy_half_adder_value_dist_11.eps', format='eps')
    plot_full_adder(inp=(0,0,0))
    plt.savefig('media/noisy_full_adder_value_dist_000.eps', format='eps')
    plot_full_adder(inp=(0,1,0))
    plt.savefig('media/noisy_full_adder_value_dist_010.eps', format='eps')
    plot_full_adder(inp=(1,1,0))
    plt.savefig('media/noisy_full_adder_value_dist_110.eps', format='eps')
    plot_full_adder(inp=(0,0,1))
    plt.savefig('media/noisy_full_adder_value_dist_001.eps', format='eps')
    plot_full_adder(inp=(0,1,1))
    plt.savefig('media/noisy_full_adder_value_dist_011.eps', format='eps')
    plot_full_adder(inp=(1,1,1))
    plt.savefig('media/noisy_full_adder_value_dist_111.eps', format='eps')
    plot_4bit_adder_dist(alpha1=0,beta1=0.05,alpha2=0,beta2=0.1)
    plt.savefig('media/noisy_4bit_adder_value_dist_full.eps', format='eps')

# plot_basic_logic()
# plt.show()
# plot_half_adder()
# plt.show()
# plot_full_adder(inp=(1,1,0))
# plt.show()
# plot_4bit_adder(alpha=0.05,beta=0.05)
# plt.show()
# plot_4bit_adder_dist(alpha1=0.02,beta1=0.02,alpha2=0.05,beta2=0.05)
# plt.show()

# print_distribution()
# check_distribution()

# gen_paper_plots()

plot_half_adder_error(beta_max=0.01)
plt.show()