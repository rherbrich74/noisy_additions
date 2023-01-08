# base.jl        Exploratory code for the (mixed)-base idea
#
# 2023 written by Ralf Herbrich
# Hasso-Plattner Institute

include("noisy.jl")

using .Noisy
using Plots
using Formatting
"""
    transform_base(x::Int; base=√2)

Returns the transformation of a binary string (as represented by the binary-representation of the integer passed) into an arbitrary base number
"""
function transform_base(x::Int; base=√2)
    # get the bis stream
    b = Vector{Int}()
    while(x > 0)
        push!(b, x % 2)
        x >>= 1
    end

    y = 0
    for i = lastindex(b):-1:firstindex(b)
        y = y*base + b[i]
    end

    return (y)
end

# x = 0b10001
# y = 0b101
# println(x, ", ", transform_base(Int(x)), ",", y)

# plot basic logic functions
gr()
rng = collect(0:(256*256-1))
plt = plot(map(x -> transform_base(x,base=2), rng), map(x -> transform_base(x,base=√2), rng), legend=false)
xlabel!(plt, "Number in Base 2", fontsize=12)
ylabel!(plt, "Number in Base √2", fontsize=12)
display(plt)
println("Press any key to continue")
readline()