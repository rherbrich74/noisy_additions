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
function plot_one_valued_logic(f; α=0)
    colors = [:red, :green, :blue, :black]
    β_range = range(0, 0.5, length=100)
    plt = plot(xlims = (0,0.5), ylims = (0,1), grid=true)
    i = 1
    for key in keys(f(α,0))
        plot_key = (Int(key[1]),Int(key[2]))
        plot!(plt, β_range, map(β -> f(α,β)[key][1], β_range), 
              linecolor=colors[i], linewidth=2.5, label="inp=$plot_key", fontsize=10, legend=:bottomright)
        i += 1
    end
    xlabel!(plt, "β", fontsize=12)
    ylabel!(plt, sprintf1("P(out=1|inp,α=%5.2f)", α), fontsize=12, )
    title!(plt, sprintf1("Probability P(1|inp) for α=%5.2f", α), fontsize=12)
    return (plt)
end

"""
    plot_two_valued_logic(f, inp=(0,0), α=0, plot_title=true)

Plots the probability distribution of a two-valued logic function `f` for a fixed input `inp` over all values of β and the value of `α`
"""
function plot_two_valued_logic(f, inp=(0,0); α=0, plot_title=true)
    colors = [:red, :green, :blue, :black]
    β_range = range(0, 0.5, length=100)
    plt = plot(xlims = (0,0.5), ylims = (0,1), grid=true)
    ks = collect(keys(f(α,0)))
    value_keys = keys(f(α,0)[ks[1]])
    i = 1
    for val in value_keys
        val_plot = (Int(val[1]),Int(val[2]))
        plot!(plt, β_range, map(β -> f(α,β)[inp][val], β_range), 
              linecolor=colors[i], linewidth=2.5, label="P($val_plot | inp=$inp)", fontsize=10, legend=:bottomright)
        i += 1
    end
    xlabel!(plt, "β", fontsize=12)
    ylabel!(plt, sprintf1("P(out|inp=$inp,α=%5.2f)", α), fontsize=12)
    if (plot_title)
        title!(plt, sprintf1("Probability Distribution for α=%5.2f and Input $inp", α), fontsize=12)
    end
    return (plt)
end

"""
    plot_two_valued_error(f_map, α=0)

Plots the error distribution for a two-valued logic function `f` over all values of β and the value of `α`
"""
function plot_two_valued_error(f; α=0, legend_pos=:bottomright)
    # compute the true output
    f_map = f(0,0)
    true_output = Dict{keytype(f_map),Tuple{UInt8,UInt8}}()
    for (k,v) in f_map
        for (k2,v2) in v
            if (v2 == 1) 
                true_output[k] = k2
                break
            end
        end
    end 

    # now plot the error rates for all the possible inputs
    colors = [:red, :green, :blue, :black, :cyan, :brown, :gray, :yellow]
    β_range = range(0, 0.5, length=100)
    plt = plot(xlims = (0,0.5), ylims = (0,1), grid=true)
    i = 1
    for key in collect(keys(f(α,0)))
        plot_key = (length(key) == 3) ? (Int(key[1]),Int(key[2]),Int(key[3])) : (Int(key[1]),Int(key[2]))
        plot!(plt, β_range, map(β -> 1 - f(α,β)[key][true_output[key]], β_range), 
              legend=legend_pos,
              ylims=(0,1),
              linecolor=colors[((i-1) % length(colors)) + 1], 
              linewidth=2.5, label="P(error|inp=$plot_key)", fontsize=10)
        i += 1
    end
    xlabel!(plt, "β", fontsize=12)
    ylabel!(plt, sprintf1("P(error|α=%5.2f)", α), fontsize=12)
    title!(plt, sprintf1("Error Rate for α=%5.2f", α), fontsize=12)
    return (plt)
end





# plot basic logic functions
gr()
anim = @animate for α ∈ range(0,0.5, length=100)
    plot_one_valued_logic(pnand, α=α) 
end
gif(anim, "anim_basic_correct_fps10.gif", fps = 10)

# plot half adder logic functions
gr()
anim = @animate for α ∈ range(0,0.5, length=100)
    plot_two_valued_error((α,β) ->
        half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), α=α)
end
gif(anim, "anim_ha_error_fps10.gif", fps = 10)

# plot full adder logic functions
gr()
anim = @animate for α ∈ range(0,0.5, length=100)
    plot_two_valued_error((α,β) ->
        full_adder(half_adder(pnand(α,β),pnor(α,β),pnot(α,β)),pnor(α,β),pnot(α,β)), α=α)
end
gif(anim, "anim_fa_error_fps10.gif", fps = 10)

# # plot half adder logic functions with psafe
# gr()
# anim = @animate for α ∈ range(0,0.5, length=100)
#     plot_two_valued_error((α,β) ->
#         psafe(half_adder(pnand(α,β),pnor(α,β),pnot(α,β)), α=α, β=β), α=α)
    
# end
# gif(anim, "anim_ha_error_with_psafe_fps10.gif", fps = 10)

# # plot full adder logic functions with psafe
# gr()
# anim = @animate for α ∈ range(0,0.5, length=100)
#     plot_two_valued_error((α,β) ->
#         psafe(full_adder(half_adder(pnand(α,β),pnor(α,β),pnot(α,β)),pnor(α,β),pnot(α,β)), α=α, β=β), α=α)
    
# end
# gif(anim, "anim_fa_error_with_psafe_fps10.gif", fps = 10)


# plot the error distribution without correction
gr()
x = 100
y = 88
α = 0
anim = @animate for β ∈ range(0,0.1, length=100)
    β_range = map(i -> β, 1:8)
    ha = map(i -> half_adder(pnand(α,β_range[i]),pnor(α,β_range[i]),pnot(α,β_range[i])),1:8) 
    fa = map(i -> full_adder(ha[i],pnor(α,β_range[i]),pnot(α,β_range[i])),1:8)
    fb = eight_bit_adder(ha[1], fa[2:8], x, y)
    bar(map(kv->kv[2], sort([(k,v) for (k,v) in fb], by=first)), legend=false)
    title!(sprintf1("Probability Distribution over $x + $y for β=%5.2f", β), fontsize=12)
    xlabel!("sum of $x + $y")
end
gif(anim, "anim_100_88_no_decay_fps10.gif", fps = 10)


# plot the error distribution with linearly decaying error rate correction
gr()
x = 100
y = 88
α = 0
anim = @animate for β ∈ range(0,0.1, length=100)
    β_range = map(i -> β/i, 1:8)
    ha = map(i -> half_adder(pnand(α,β_range[i]),pnor(α,β_range[i]),pnot(α,β_range[i])),1:8) 
    fa = map(i -> full_adder(ha[i],pnor(α,β_range[i]),pnot(α,β_range[i])),1:8)
    fb = eight_bit_adder(ha[1], fa[2:8], x, y)
    bar(map(kv->kv[2], sort([(k,v) for (k,v) in fb], by=first)), legend=false)
    title!(sprintf1("Probability Distribution over $x + $y for β=%5.2f", β), fontsize=12)
    xlabel!("sum of $x + $y")
end
gif(anim, "anim_100_88_linear_decay_fps10.gif", fps = 10)

# plot the error distribution with redundancy correction
gr()
x = 100
y = 88
α = 0
anim = @animate for β ∈ range(0,0.1, length=100)
    ha = Vector{Dict{Tuple{UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}}}()
    fa = Vector{Dict{Tuple{UInt8,UInt8,UInt8},Dict{Tuple{UInt8,UInt8},Float64}}}()
    push!(ha, half_adder(pnand(α,β),pnor(α,β),pnot(α,β)))
    push!(fa, full_adder(ha[1],pnor(α,β),pnot(α,β)))
    for i = 2:8
        push!(ha, psafe(ha[i-1], α=α, β=β))
        push!(fa, psafe(fa[i-1], α=α, β=β))
    end
    fb = eight_bit_adder(ha[1], fa[2:8], x, y)
    bar(map(kv->kv[2], sort([(k,v) for (k,v) in fb], by=first)), legend=false)
    title!(sprintf1("Probability Distribution over $x + $y for β=%5.2f", β), fontsize=12)
    xlabel!("sum of $x + $y")
end
gif(anim, "anim_100_88_linear_psafe_decay_fps10.gif", fps = 10)

