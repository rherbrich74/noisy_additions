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
    pnand(α, β)

Returns a dictionary that computes the probability distribution of a probabilistic NAND gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnand(α, β)
    return (Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}}(
        (0,0)   => Dict{UInt8,Float64}(0 => α*α,                1 => 1 - α*α),
        (0,1)   => Dict{UInt8,Float64}(0 => α*(1 - β),          1 => 1 - α*(1 - β)),
        (1,0)   => Dict{UInt8,Float64}(0 => α*(1 - β),          1 => 1 - α*(1 - β)),
        (1,1)   => Dict{UInt8,Float64}(0 => (1 - β)*(1 - β),    1 => 1 - (1 - β)*(1 - β))
    ))
end

"""
    pnor(α, β)

Returns a dictionary that computes the probability distribution of a probabilistic NOR gate with error probablisities α and β (both of which should be ∈[0,1])
    """
function pnor(α, β)
    return (Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}}(
        (0,0)   => Dict{UInt8,Float64}(0 => 1 - (1 - α)*(1 - α),    1 => (1 - α)*(1 - α)),
        (0,1)   => Dict{UInt8,Float64}(0 => 1 - (1 - α)*β,          1 => (1 - α)*β),
        (1,0)   => Dict{UInt8,Float64}(0 => 1 - (1 - α)*β,          1 => (1 - α)*β),
        (1,1)   => Dict{UInt8,Float64}(0 => 1 - β*β,                1 => β*β)
    ))
end

"""
    pnot(α, β)

Returns a dictionary that computes the probability distribution of a probabilistic NOT gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnot(α, β)
    return (Dict{UInt8,Dict{UInt8,Float64}}(
        0   => Dict{UInt8,Float64}(0 => α,      1 => 1-α),
        1   => Dict{UInt8,Float64}(0 => 1 - β,  1 => β)
    ))
end

"""
    half_adder(nand, nor, not)

Returns a dictionary that computes the probability distribution of a probabilistic half-adder (based on probabilistic `nand`, `nor` and `not` gates)
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
    full_adder(ha, nor, not)

Returns a dictionary that computes the probability distribution of a probabilistic full-adder (based on probabilistic `ha` (half-adder), `nor` and `not` gates)
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
    four_bit_adder(ha, fa, A, B)

Returns a dictionary that computes the probability distribution of a probabilistic 4bit-adder for the two inputs `A` and `B` (based on probabilistic `ha` (half-adder), `fa` (full-adder) gates).
"""
function four_bit_adder(ha::Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}},
                        fa::Dict{Tuple{UInt8,UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}},
                        A, B)::Dict{UInt8,Float64}
    a0 = A & 1; A >>= 1
    a1 = A & 1; A >>= 1
    a2 = A & 1; A >>= 1
    a3 = A & 1
    b0 = B & 1; B >>= 1
    b1 = B & 1; B >>= 1
    b2 = B & 1; B >>= 1
    b3 = B & 1
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
    return(p_dict)
end

"""
    four_bit_adder(ha, fa)

Returns a dictionary that computes the probability distribution of a probabilistic 4bit-adder (based on probabilistic `ha` (half-adder), `fa` (full-adder) gates).
"""
function four_bit_adder(ha::Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}},
                        fa::Dict{Tuple{UInt8,UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}})
    D = Dict{Tuple{UInt8,UInt8},Dict{UInt8,Float64}}()
    for inp = 0:255
        A = inp & 15
        B = (inp >> 4) & 15
        D[(A,B)] = four_bit_adder(ha, fa, A, B)
    end
    return(D)
end

"""
    eight_bit_adder(ha, fa, A, B)

Returns a dictionary that computes the probability distribution of a probabilistic 8bit-adder for the two inputs `A` and `B` (based on probabilistic `ha` (half-adder), `fa` (full-adder) gates).
"""
function eight_bit_adder(ha::Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}},
                         fa::Vector{Dict{Tuple{UInt8,UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}}},
                         A, B)::Dict{UInt16,Float64}
    a0 = A & 1; A >>= 1
    a1 = A & 1; A >>= 1
    a2 = A & 1; A >>= 1
    a3 = A & 1; A >>= 1
    a4 = A & 1; A >>= 1
    a5 = A & 1; A >>= 1
    a6 = A & 1; A >>= 1
    a7 = A & 1
    b0 = B & 1; B >>= 1
    b1 = B & 1; B >>= 1
    b2 = B & 1; B >>= 1
    b3 = B & 1; B >>= 1
    b4 = B & 1; B >>= 1
    b5 = B & 1; B >>= 1
    b6 = B & 1; B >>= 1
    b7 = B & 1
    p_dict = Dict{UInt16,Float64}()
    for out = 0:511
        S = out
        s0 = out & 1; out >>= 1
        s1 = out & 1; out >>= 1
        s2 = out & 1; out >>= 1
        s3 = out & 1; out >>= 1
        s4 = out & 1; out >>= 1
        s5 = out & 1; out >>= 1
        s6 = out & 1; out >>= 1
        s7 = out & 1; out >>= 1
        s8 = out & 1
        p = 0
        for marginal = 0:127
            # extract the invidiual bits
            c0 = marginal & 1; marginal >>= 1
            c1 = marginal & 1; marginal >>= 1
            c2 = marginal & 1; marginal >>= 1
            c3 = marginal & 1; marginal >>= 1
            c4 = marginal & 1; marginal >>= 1
            c5 = marginal & 1; marginal >>= 1
            c6 = marginal & 1
            p += ha[(a0,b0)][(s0,c0)] * fa[1][(a1,b1,c0)][(s1,c1)] * fa[2][(a2,b2,c1)][(s2,c2)] * 
                 fa[3][(a3,b3,c2)][(s3,c3)] * fa[4][(a4,b4,c3)][(s4,c4)] * fa[5][(a5,b5,c4)][(s5,c5)] * 
                 fa[6][(a6,b6,c5)][(s6,c6)] * fa[7][(a7,b7,c6)][(s7,s8)]
        end
        p_dict[S] = p
    end
    return(p_dict)
end

"""
    check_distribution(p_map)

Returns true, if the distribution object is a true normalizing distribution for all keys of `p_map`
"""
function check_distribution(p_map=full_adder(half_adder(pnand(0,0),pnor(0,0),pnot(0,0)),pnor(0,0),pnot(0,0)))
    for v in values(p_map)
        if (abs(sum(values(v))-1.0) > 1e-4)
            return (false)
        end
    end
    return(true)
end

"""
    print_distribution(p, title="")

Prints the probabilistic map `p` on the screen with title line `title`
"""
function print_distribution(p_map=half_adder(pnand(0,0),pnor(0,0),pnot(0,0)), title="")
    if (length(title) > 0)
        println(title)
    end
    for (k,v) in p_map
        println(k, ": ", v)
    end
end

#################################################################
# Basic plotting code
#################################################################

"""
    plot_one_valued_logic(f, α=0)

Plots the probability distribution of a one-valued logic function `f` over all values of β and the value of `α`
"""
function plot_one_valued_logic(f,α=0)
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
    title!(plt, sprintf1("Probability P(1|inp) for α=%5.2f", α), fontsize=12)
    return (plt)
end

"""
    plot_two_valued_logic(f, inpu=(0,0), α=0, plot_title=true)

Plots the probability distribution of a two-valued logic function `f` for a fixed input `inp` over all values of β and the value of `α`
"""
function plot_two_valued_logic(f, inp=(0,0), α=0, plot_title=true)
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
    ylabel!(plt, sprintf1("P(out|inp=$inp,α=%5.2f)", α), fontsize=12)
    if (plot_title)
        title!(plt, sprintf1("Probability Distribution for α=%5.2f and Input $inp", α), fontsize=12)
    end
    return (plt)
end



# # plot basic logic functions
# gr()
# plot_one_valued_logic(pnand, 0.1) |> display 
# readline()

# # plot half adder logic functions
# gr()
# p1 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,0))
# p2 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,0))
# p3 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,1))
# p4 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,1))
# plot(p1, p2, p3, p4, layout=(2,2), size=(1200,1000)) |> display
# readline()

# # plot half adder logic functions
# gr()
# anim = @animate for α ∈ range(0,0.5, length=100)
#     p1 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,0), α, false)
#     p2 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,0), α, false)
#     p3 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,1), α, false)
#     p4 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,1), α, false)
#     plot(p1, p2, p3, p4, layout=(2,2), size=(1200,1000))
# end
# gif(anim, "anim_fps15.gif", fps = 15)

# # prints and checks the probability distributions on screen
# println(check_distribution(full_adder(half_adder(pnand(0.1,0.2),pnor(0.1,0.2),pnot(0.1,0.2)),pnor(0.1,0.2),pnot(0.1,0.2))))
# print_distribution()

gr()
x = 254
y = 1
α = 0
anim = @animate for β ∈ range(0,0.5, length=100)
    β_range = map(i -> β/i, 1:8)
    ha = map(i -> half_adder(pnand(α,β_range[i]),pnor(α,β_range[i]),pnot(α,β_range[i])),1:8) 
    fa = map(i -> full_adder(ha[i],pnor(α,β_range[i]),pnot(α,β_range[i])),1:8)
    fb = eight_bit_adder(ha[1], fa[2:8], x, y)
    bar(map(kv->kv[2], sort([(k,v) for (k,v) in fb], by=first)), legend=false)
    title!(sprintf1("Probability Distribution over $x + $y for β=%5.2f", β), fontsize=12)
    xlabel!("sum of $x + $y")
end
gif(anim, "anim_254_1_fps15.gif", fps = 15)

