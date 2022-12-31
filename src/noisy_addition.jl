# noisy_addition.jl         A set of functions to compute the output probabilities of 
#                           noisy half-, full- and 4-bit-adders
#
# 2022 written by Ralf Herbrich
# Hasso-Plattner Institute

"""
    pnand(α,β)

Returns a function that computes the probability of true for of a probabilistic NAND gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnand(α,β)
    p_00 = 1.0 - α*α
    p_01 = 1.0 - α*(1.0 - β)
    p_11 = 1.0 - (1.0 - β)*(1.0 - β)
    function (a,b)
        if (a == 0 && b == 0) 
            return (Dict{Int,Float32}(0 => 1.0 - p_00, 1 => p_00))
        elseif ((a == 0 && b == 1) || (a == 1 && b == 0)) 
            return (Dict{Int,Float32}(0 => 1.0 - p_01, 1 => p_01))
        else 
            return (Dict{Int,Float32}(0 => 1.0 - p_11, 1 => p_11))
        end
    end
end

"""
    pnor(α,β)

Returns a function that computes the probability of true for of a probabilistic NOR gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnor(α,β)
    p_00 = (1.0 - α)*(1.0 - α)
    p_01 = (1.0 - α)*β
    p_11 = β*β
    function (a,b)
        if (a == 0 && b == 0) 
            return (Dict{Int,Float32}(0 => 1.0 - p_00, 1 => p_00))
        elseif ((a == 0 && b == 1) || (a == 1 && b == 0)) 
            return (Dict{Int,Float32}(0 => 1.0 - p_01, 1 => p_01))
        else 
            return (Dict{Int,Float32}(0 => 1.0 - p_11, 1 => p_11))
        end
    end
end

"""
    pnot(α,β)

Returns a function that computes the probability of true for of a probabilistic NOT gate with error probablisities α and β (both of which should be ∈[0,1])
"""
function pnot(α,β)
    function (a)
        if (a == 0) 
            return (Dict{Int,Float32}(0 => α, 1 => 1.0 - α))
        else 
            return (Dict{Int,Float32}(0 => 1.0 - β, 1 => β))
        end
    end
end

"""
    half_adder(nand,nor,not)

Returns a function that computes the probability of true for of a probabilistic half adder (based on probabilistic `nand`, `nor` and `not` gates.
"""
function half_adder(nand,nor,not)
    # Computes the marginal distribution over sum and carry over bit given the inputs `a` and `b`
    function marginalize(a,b)
        nand_ab = nand(a,b)
        nor_ab = nor(a,b)

        D = Dict{Tuple{Int,Int},Float32}()
        for s = 0:1
            for c_out = 0:1
                p = 0.0
                for d = 0:1
                    not_d = not(d)
                    for e = 0:1
                        not_e = not(e)
                        for f = 0:1
                            nand_ef = nand(e,f)
                            for g = 0:1
                                not_g = not(g)
                                p = p + nand_ab[e] * nor_ab[d] * not_e[c_out] * not_d[f] * nand_ef[g] * not_g[s]
                            end
                        end
                    end
                end
                D[(s,c_out)] = p
            end
        end
        return(D)
    end

    function (a,b)
        if (a == 0 && b == 0) 
            return (marginalize(0,0))
        elseif ((a == 0 && b == 1) || (a == 1 && b == 0)) 
            return (marginalize(0,1))
        else 
            return (marginalize(1,1))
        end
    end
end

"""
    full_adder(ha,nor,not)

Returns a function that computes the probability of true for of a probabilistic full adder (based on probabilistic `ha` (half-adder), `nor` and `not` gates.
"""
function full_adder(ha,nor,not)
    # Computes the marginal distribution over sum and carry over bit given the inputs `a` and `b`
    function marginalize(a,b,c_in)
        ha_ab = ha(a,b)
        nor_ab = nor(a,b)

        D = Dict{Tuple{Int,Int},Float32}()
        for s = 0:1
            for c_out = 0:1
                p = 0.0
                for d = 0:1
                    ha_dc_in = ha(d,c_in)
                    for e = 0:1
                        for f = 0:1
                            nor_ef = nor(e,f)
                            for g = 0:1
                                not_g = not(g)
                                p = p + ha_ab[(d,e)] * ha_dc_in[(s,f)] * nor_ef[g] * not_g[c_out] 
                            end
                        end
                    end
                end
                D[(s,c_out)] = p
            end
        end
        return(D)
    end

    function (a,b,c_in)
        if (a == 0 && b == 0 && c_in == 0) 
            return (marginalize(0,0,0))
        elseif (a == 0 && b == 1 && c_in == 0)
            return (marginalize(0,1,0))
        elseif (a == 1 && b == 0 && c_in == 0)
            return (marginalize(1,0,0))
        elseif (a == 1 && b == 1 && c_in == 0)
            return (marginalize(1,1,0))
        elseif (a == 0 && b == 0 && c_in == 1) 
            return (marginalize(0,0,1))
        elseif (a == 0 && b == 1 && c_in == 1)
            return (marginalize(0,1,1))
        elseif (a == 1 && b == 0 && c_in == 1)
            return (marginalize(1,0,1))
        elseif (a == 1 && b == 1 && c_in == 1)
            return (marginalize(1,1,1))
        end
    end
end


α = 0
total_time = @time for β in 0.0:1e-4:1.0
    β = 0.1
    nand_map = pnand(α,β)
    nor_map = pnor(α,β)
    not_map = pnot(α,β)
    ha_map = half_adder(nand_map,nor_map,not_map) 

    # println(ha_map(0,0))
    # println(ha_map(0,1))
    # println(ha_map(1,0))
    # println(ha_map(1,1))

    fa_map = full_adder(ha_map,nor_map,not_map)
    fa_000 = fa_map(0,0,0)
end

println("Elapsed time = ", total_time)