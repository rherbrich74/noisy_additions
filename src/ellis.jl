# ellis.jl         All plotting code for the ELLIS workshop presentation
#
# 2023 written by Ralf Herbrich
# Hasso-Plattner Institute

include("noisy.jl")

using .Noisy
using Plots
using Formatting

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

# plot half adder logic functions
gr()
p1 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,0))
p2 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,0))
p3 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (0,1))
p4 = plot_two_valued_logic((α,β) -> half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), (1,1))
plot(p1, p2, p3, p4, layout=(2,2), size=(1200,1000)) |> display
readline()

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

# gr()
# x = 254
# y = 1
# α = 0
# anim = @animate for β ∈ range(0,0.5, length=100)
#     β_range = map(i -> β/i, 1:8)
#     ha = map(i -> half_adder(pnand(α,β_range[i]),pnor(α,β_range[i]),pnot(α,β_range[i])),1:8) 
#     fa = map(i -> full_adder(ha[i],pnor(α,β_range[i]),pnot(α,β_range[i])),1:8)
#     fb = eight_bit_adder(ha[1], fa[2:8], x, y)
#     bar(map(kv->kv[2], sort([(k,v) for (k,v) in fb], by=first)), legend=false)
#     title!(sprintf1("Probability Distribution over $x + $y for β=%5.2f", β), fontsize=12)
#     xlabel!("sum of $x + $y")
# end
# gif(anim, "anim_254_1_fps15.gif", fps = 15)

