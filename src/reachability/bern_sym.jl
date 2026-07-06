@with_kw struct BernSym <: NV.Solver
    truncation_terms = 50
    separate_relaxations = true
    relaxations = :shift
    splitting_depth = 0
    # set to true in beginning to get first α
    init = false
    init_method = :CROWNQuad
    save_bounds = false
    common_generators = false
    use_shortcut = true
    use_memory_optimizations = true
end


function forward_linear(
    solver::BernSym,
    L::NV.LayerNegPosIdx,
    input::BernsteinInterval,
)
    if solver.common_generators
        error("unimplemented")
    else
        out = interval_map(
            L.W_neg,
            L.W_pos,
            input,
            L.bias,
        )
    end
    return out
end
function forward_linear(
    solver::BernSym,
    L::NV.LayerNegPosIdx,
    input::CombinedPolyBernsteinInterval,
)
    if solver.common_generators
        error("unimplemented")
    else
        out = interval_map(
            L.W_neg,
            L.W_pos,
            input,
            L.bias,
        )
    end
    return out
end
function forward_linear(solver::BernSym, L::CROWNLayer, input::BernsteinInterval)
    if solver.common_generators
        error("unimplemented")
        #Low, Up = interval_map_common(
        #    min.(0, L.weights),
        #    max.(0, L.weights),
        #    input.poly_interval.Low,
        #    input.poly_interval.Up,
        #    L.bias,
        #)
    else
         return interval_map(
            min.(0, L.weights),
            max.(0, L.weights),
            input,
            L.bias;
            use_memory_optimizations=solver.use_memory_optimizations
        )
    end
end
function forward_linear(solver::BernSym, L::CROWNLayer, input::CombinedPolyBernsteinInterval)
    if solver.common_generators
        error("unimplemented")
        #Low, Up = interval_map_common(
        #    min.(0, L.weights),
        #    max.(0, L.weights),
        #    input.poly_interval.Low,
        #    input.poly_interval.Up,
        #    L.bias,
        #)
    else
         return interval_map(
            min.(0, L.weights),
            max.(0, L.weights),
            input,
            L.bias;
            use_memory_optimizations=solver.use_memory_optimizations
        )
    end
end



function forward_act(
    solver::BernSym,
    L::Union{NV.LayerNegPosIdx{NV.Id},CROWNLayer{NV.Id}},
    input::Union{BernsteinInterval,CombinedPolyBernsteinInterval} ,
    α,
)
    return input
end

function forward_act(
    solver::BernSym,
    L::Union{NV.LayerNegPosIdx{NV.ReLU} , CROWNLayer{NV.ReLU}},
    input::Union{BernsteinInterval,CombinedPolyBernsteinInterval},
    α,
)
    sym = input

    #    error("unimplemented")
    #    s = truncate_desired_common(sym, solver.truncation_terms)
    #else
    #    s = truncate_desired(sym, solver.truncation_terms)
    #end
    s = sym

    if s isa BernsteinInterval 
	ll, lu = bounds(s.Low,s.X)
	ul, uu = bounds(s.Up,s.X)
    elseif s isa CombinedPolyBernsteinInterval  
	ll ,lu ,ul, uu = bounds(s)
    else 
	throw("should be one of these two types")
    end 

    if solver.save_bounds
        throw("unimplemented")
        #ChainRulesCore.ignore_derivatives() do
        #    input.lbs[L.index] .= max.(ll, input.lbs[L.index])
        #    input.ubs[L.index] .= min.(uu, input.ubs[L.index])
        #end
    end

    #if solver.init
        if solver.init_method == :Chebyshev
            # Chebyshev initialisation
            resl = relax_relu_chebyshev.(ll, lu, 2*ones(Integer, n))
            resu = relax_relu_chebyshev.(ul, uu, 2*ones(Integer, n))
            α[1, 1:2, :] .= vecOfVec2Mat(first.(resl))'[2:3, :]
            α[2, 1:2, :] .= vecOfVec2Mat(first.(resu))'[2:3, :]

            cₗ = vecOfVec2Mat(first.(resl))
            cₗ[:, 1] .-= last.(resl)
            cᵤ = vecOfVec2Mat(first.(resu))
            cᵤ[:, 1] .+= last.(resu)

            cₗ = [c for c in eachrow(cₗ)]
            cᵤ = [c for c in eachrow(cᵤ)]
        elseif solver.init_method == :CROWNQuad
            # CROWNQuad initialisation
            cₗ = relax_relu_crown_quad_lower.(ll, lu)
            cᵤ = relax_relu_crown_quad_upper.(ul, uu)

            # CROWNQuad is quadratic relaxation, so set first two params
            α[1, 1:2, :] .= vecOfVec2Mat(cₗ)'[2:3, :]
            α[2, 1:2, :] .= vecOfVec2Mat(cᵤ)'[2:3, :]

            cₗ = vecOfVec2Mat(cₗ)
            cᵤ = vecOfVec2Mat(cᵤ)
        else
            throw(ArgumentError("Initialisation method $(solver.init_method) not known!"))
        end
    #else
        #cₗ = [ifelse(l >= 0, [0., 1, 0], ifelse(u <= 0, zeros(3), get_lower_polynomial_shift(l, u, 2, a))) for (l, u, a) in zip(ll, lu, eachcol(α[1,:,:]))]
        #cᵤ = [ifelse(l >= 0, [0., 1, 0], ifelse(u <= 0, zeros(3), get_upper_polynomial_shift(l, u, 2, a))) for (l, u, a) in zip(ul, uu, eachcol(α[2,:,:]))]
        #cₗ = get_lower_polynomial_shift.(ll, lu, 2, eachcol(α[1,:,:]))
        #cᵤ = get_upper_polynomial_shift.(ul, uu, 2, eachcol(α[2,:,:]))
        cₗ = get_lower_polynomial_shift(ll, lu, 2, α[1, :, :]')
        cᵤ = get_upper_polynomial_shift(ul, uu, 2, α[2, :, :]')
    #end

    #cₗ = vecOfVec2Mat(cₗ)
    #cᵤ = vecOfVec2Mat(cᵤ)

    if solver.common_generators
        throw("unimplemented")
        #L̂, Û = quad_prop_common(cₗ, cᵤ, s.Low, s.Up, ll, lu, ul, uu)
    else
        #L̂ = fast_quad_prop(cₗ[:, 3], cₗ[:, 2], cₗ[:, 1], s.Low, ll, lu)
        #Û = fast_quad_prop(cᵤ[:, 3], cᵤ[:, 2], cᵤ[:, 1], s.Up, ul, uu)
	if s isa BernsteinInterval
	    L̂ = quadratic_propagation(cₗ[:, 3], cₗ[:, 2], cₗ[:, 1], s.Low )
	    Û = quadratic_propagation(cᵤ[:, 3], cᵤ[:, 2], cᵤ[:, 1], s.Up )
	elseif s isa CombinedPolyBernsteinInterval  
	    ŝ = quadratic_propagation(cₗ[:, 3], cₗ[:, 2], cₗ[:, 1],cᵤ[:, 3], cᵤ[:, 2], cᵤ[:, 1], s)
	else 
	    throw("should be one of these two types")
	end 
    end

    if s isa BernsteinInterval 
	return BernsteinInterval(L̂, Û, input.n,input.X)
    elseif s isa CombinedPolyBernsteinInterval  
	return ŝ 
    else 
	throw("should be one of these two types")
    end 
end

function forward_network(solver::BernSym, net::Chain, input::CombinedPolyBernsteinInterval)
    degree = 2
    α0 = initialize_params(net, degree, method = :zero)
    # convert flat params to per-layer (2,degree,n) arrays
    αs = vec2propagation(net, degree, α0)

    out = input
    for (L, α) in zip(net.layers, αs)
        out = forward_linear(solver, L, out)
        out = forward_act(solver, L, out, α)
    end
    return out
end

"""
Initialize the symbolic domain corresponding to the given solver with the respective input set.
"""
function initialize_symbolic_domain(
    solver::BernSym,
    net,
    input::AbstractHyperrectangle; use_combined_repr=true
)
    if use_combined_repr
	return init_combined_bernstein_interval(input) 
    end
    return init_bernstein_interval(input)
end
