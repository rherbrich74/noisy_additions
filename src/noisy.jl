# noisy.jl         A set of functions to compute the output probabilities of 
#                  noisy half-, full-, 4-bit-adders and 8-bit adders
#
# 2023 written by Ralf Herbrich
# Hasso-Plattner Institute

module Noisy

export pnand, pnor, pnot, half_adder, full_adder, four_bit_adder, eight_bit_adder, check_distribution, print_distribution

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

end