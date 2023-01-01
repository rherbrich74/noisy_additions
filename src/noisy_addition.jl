# noisy_addition.jl         A set of functions to compute the output probabilities of 
#                           noisy half-, full-, 4-bit-adders and 6-bit adders
#
# 2023 written by Ralf Herbrich
# Hasso-Plattner Institute

using Plots
using Formatting

#################################################################
# Basic message passing code
#################################################################

"""
    pnand(α,β)

Returns a dictionary that computes the probability distribution of a probabilistic NAND gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnand(α,β)
    return (Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}}(
        (0,0)   => Dict{UInt8,Float64}(0 => α*α,                1 => 1 - α*α),
        (0,1)   => Dict{UInt8,Float64}(0 => α*(1 - β),          1 => 1 - α*(1 - β)),
        (1,0)   => Dict{UInt8,Float64}(0 => α*(1 - β),          1 => 1 - α*(1 - β)),
        (1,1)   => Dict{UInt8,Float64}(0 => (1 - β)*(1 - β),    1 => 1 - (1 - β)*(1 - β))
    ))
end

"""
    pnor(α,β)

Returns a dictionary that computes the probability distribution of a probabilistic NOR gate with error probablisities α and β (both of which should be ∈[0,1])
    """
function pnor(α,β)
    return (Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}}(
        (0,0)   => Dict{UInt8,Float64}(0 => 1 - (1 - α)*(1 - α),    1 => (1 - α)*(1 - α)),
        (0,1)   => Dict{UInt8,Float64}(0 => 1 - (1 - α)*β,          1 => (1 - α)*β),
        (1,0)   => Dict{UInt8,Float64}(0 => 1 - (1 - α)*β,          1 => (1 - α)*β),
        (1,1)   => Dict{UInt8,Float64}(0 => 1 - β*β,                1 => β*β)
    ))
end

"""
    pnot(α,β)

Returns a dictionary that computes the probability distribution of a probabilistic NOT gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnot(α,β)
    return (Dict{UInt8,Dict{UInt8,Float64}}(
        0   => Dict{UInt8,Float64}(0 => α,      1 => 1-α),
        1   => Dict{UInt8,Float64}(0 => 1 - β,  1 => β)
    ))
end

"""
    half_adder(nand,nor,not)

Returns a dictionary that computes the probability distribution of a probabilistic half-adder (based on probabilistic `nand`, `nor` and `not` gates).
"""
function half_adder(nand::Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}},
                    nor::Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}},
                    not::Dict{UInt8,Dict{UInt8,Float64}})
    D = Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}}()
    for inp = 0:3
        b = inp & 1; inp >>= 1
        a = inp & 1
        p_dict = Dict{Tuple{UInt8,UInt8},Float64}()
        for out = 0:3
            c = out & 1; out >>= 1
            s = out & 1
            p = 0
            for marginal = 0:15
                # extract the invidiual bits
                g = marginal & 1; marginal >>= 1
                f = marginal & 1; marginal >>= 1
                e = marginal & 1; marginal >>= 1
                d = marginal & 1
                p += nand[(a,b)][e] * nor[(a,b)][d] * not[e][c] * not[d][f] * nand[(e,f)][g] * not[g][s]
            end
            p_dict[(s,c)] = p
        end
        D[(a,b)] = p_dict
    end
    return(D)
end

"""
    full_adder(ha,nor,not)

Returns a dictionary that computes the probability distribution of a probabilistic full-adder (based on probabilistic `ha` (half-adder), `nor` and `not` gates).
"""
function full_adder(ha::Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}},
                    nor::Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}},
                    not::Dict{UInt8,Dict{UInt8,Float64}})
    D = Dict{Tuple{UInt8,UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}}()
    for inp = 0:7
        b = inp & 1; inp >>= 1
        a = inp & 1; inp >>= 1
        c_in = inp & 1
        p_dict = Dict{Tuple{UInt8,UInt8},Float64}()
        for out = 0:3
            c_out = out & 1; out >>= 1
            s = out & 1
            p = 0
            for marginal = 0:15
                # extract the invidiual bits
                g = marginal & 1; marginal >>= 1
                f = marginal & 1; marginal >>= 1
                e = marginal & 1; marginal >>= 1
                d = marginal & 1
                p += ha[(a,b)][(d,e)] * ha[(d,c_in)][(s,f)] * nor[(e,f)][g] * not[g][c_out] 
            end
            p_dict[(s,c_out)] = p
        end
        D[(a,b,c_in)] = p_dict
    end
    return(D)
end

"""
    full_adder(ha,nor,not)

Returns a dictionary that computes the probability distribution of a probabilistic 4bit-adder (based on probabilistic `ha` (half-adder), `fa` (full-adder) gates).
"""
function four_bit_adder(ha::Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}},
                        fa::Dict{Tuple{UInt8,UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}})
    D = Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}}()
    for inp = 0:255
        A = inp & 15
        B = (inp >> 4) & 15
        a0 = inp & 1; inp >>= 1
        a1 = inp & 1; inp >>= 1
        a2 = inp & 1; inp >>= 1
        a3 = inp & 1; inp >>= 1
        b0 = inp & 1; inp >>= 1
        b1 = inp & 1; inp >>= 1
        b2 = inp & 1; inp >>= 1
        b3 = inp & 1
        p_dict = Dict{UInt8,Float64}()
        for out = 0:31
            S = out
            s0 = out & 1; out >>= 1
            s1 = out & 1; out >>= 1
            s2 = out & 1; out >>= 1
            s3 = out & 1; out >>= 1
            s4 = out & 1
            p = 0
            for marginal = 0:7
                # extract the invidiual bits
                c0 = marginal & 1; marginal >>= 1
                c1 = marginal & 1; marginal >>= 1
                c2 = marginal & 1
                p += ha[(a0,b0)][(s0,c0)] * fa[(a1,b1,c0)][(s1,c1)] * fa[(a2,b2,c1)][(s2,c2)] * fa[(a3,b3,c2)][(s3,s4)]
            end
            p_dict[S] = p
        end
        D[(A,B)] = p_dict
    end
    return(D)
end

#################################################################
# Basic plotting code
#################################################################

"""
    plot_one_valued_logic(α)

Plots the probability distribution of a one-valued logic function
"""
function plot_one_valued_logic(f, α=0)
    colors = [:red, :green, :blue, :black]
    β_range = range(0, 0.5, length=100)
    plt = plot(xlims = (0,0.5), ylims = (0,1), grid=true)
    i = 1
    for key in keys(f(α,0))
        plot!(plt, β_range, map(β -> f(α,β)[key][1], β_range), linecolor=colors[i], linewidth=2.5, label="inp=$key", fontsize=10)
        i += 1
    end
    xlabel!(plt, "β", fontsize=12)
    ylabel!(plt, sprintf1("P(out=1|inp,α=%5.2f)", α), fontsize=12, )
    title!(plt, sprintf1("Probability Distribution for α=%5.2f", α), fontsize=12)
    return (plt)
end

"""
plot_two_valued_logic(α)

Plots the probability distribution of a two-valued logic function for a fixed input
"""
function plot_two_valued_logic(f, inp=(0,0), α=0)
    colors = [:red, :green, :blue, :black]
    β_range = range(0, 0.5, length=100)
    plt = plot(xlims = (0,0.5), ylims = (0,1), grid=true)
    ks = collect(keys(f(α,0)))
    value_keys = keys(f(α,0)[ks[1]])
    i = 1
    for val in value_keys
        plot!(plt, β_range, map(β -> f(α,β)[inp][val], β_range), linecolor=colors[i], linewidth=2.5, label="P($val | inp=$inp)", fontsize=10)
        i += 1
    end
    xlabel!(plt, "β", fontsize=12)
    ylabel!(plt, sprintf1("P(out|inp=$inp,α=%5.2f)", α), fontsize=12, )
    title!(plt, sprintf1("Probability Distribution for α=%5.2f and Input $inp", α), fontsize=12)
    return (plt)
end



gr()
# display(plot_one_valued_logic(pnot, 0.1))
# plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,0)) |> dislpay
# plot_two_valued_logic((α,β) -> full_adder(half_adder(pnand(α,β),pnor(α,β),pnot(α,β)),pnor(α,β),pnot(α,β)), (0,0,1)) |> display
# readline()

anim = @animate for α ∈ range(0,0.5, length=100)
    plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,1), α)
end
gif(anim, "anim_fps15.gif", fps = 15)

# function print_map(m, s)
#     println(s)
#     for (k,v) in m
#         println(k, ": ", v)
#     end
# end

# α = 0
# β = 0
# nand_map = pnand(α,β)
# nor_map = pnor(α,β)
# not_map = pnot(α,β)
# ha_map = half_adder(nand_map,nor_map,not_map) 
# fa_map = full_adder(ha_map,nor_map,not_map) 
# fb_map = four_bit_adder(ha_map,fa_map) 

# print_map(nand_map, "NAND")
# print_map(nor_map,  "NOR")
# print_map(not_map,  "NOT")
# print_map(ha_map,   "Half Adder")
# print_map(fa_map,   "Full Adder")
# print_map(fb_map,   "4-bit Adder")

# α = 0
# @time for β in 0.0:1e-4:1.0
#     β = 0.1
#     nand_map = pnand(α,β)
#     nor_map = pnor(α,β)
#     not_map = pnot(α,β)
#     ha_map = half_adder(nand_map,nor_map,not_map) 
#     fa_map = full_adder(ha_map,nor_map,not_map)
#     fb_map = four_bit_adder(ha_map,fa_map)
# end
