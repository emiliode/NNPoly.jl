using LazySets
using DynamicPolynomials


struct CombinedPolyBernsteinInterval{N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}, VN<:AbstractArray{N}}
    Low::TN # each row of the matrix corresponds to the term coefficients for one Low polynomial
    Up::TN # each row of the matrix corresponds to the term coefficients for one Upper polynomial
    bern_terms::TN # matrix containing the bernstein terms without coefficients. 
    t::M # number of terms
    n::M # input dimension
    orders::Vector{Int64} # orders of the variables
    X::Hyperrectangle{N}
    lbs::Vector{VN}  # (vector of vectors) lower bounds for intermediate values
    ubs::Vector{VN}  # (vector of vectors) upper bounds for intermediate values
end

function CombinedPolyBernsteinInterval(Low::TN, Up::TN, bern_terms::TN, t::M, n::M, orders::Vector{Int64},X::Hyperrectangle{N}) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    return CombinedPolyBernsteinInterval(Low, Up,bern_terms, t, n, orders, X, Vector{Vector{N}}(), Vector{Vector{N}}())
end

"""
Creates a CombinedMultiBernsteinImp from the given multivariate polynomials.
"""
function make_polynomial(
    polynoms::Vector{<:AbstractPolynomialLike},
    orders::Vector{Int64},
    X::Hyperrectangle{T},
) where {T<:Number}
    basis_vector = get_basis_vectors(orders, X)
    all_bern_coeff = []
    C =  zeros(Float64, length(polynoms), 0)   # num_polys×0 matrix
    # get all the terms and the Coefficient matrix
    all_terms = []
    for (poly_idx,polynom) in enumerate(polynoms)
	for t in terms(polynom)
	    t_idx = findfirst(x -> x == t,all_terms)
	    if isnothing(t_idx)
		push!(all_terms,t)
		C = [C zeros(eltype(C),size(C,1))]
		t_idx = length(all_terms)
	    end
	    C[poly_idx,t_idx] +=  DynamicPolynomials.coefficient(t)
	end
	
    end
    for t in all_terms
	ev = exponents(t)
	for (i, e) in enumerate(ev)
	    a = center(X)[i] - radius_hyperrectangle(X, i)
	    b = center(X)[i] + radius_hyperrectangle(X, i)
	    bern_coeff = calculate_bern_coeff_for_monomial(e, orders[i], a, b)
	    bern_coeff = [bern_coeff; zeros(maximum(orders)-orders[i])]
	    all_bern_coeff = push!(all_bern_coeff, bern_coeff)
	end
    end
    bern_terms_matrix = permutedims(stack(all_bern_coeff))
    return CombinedMultiBernsteinImp(
	C,
	copy(C),
        bern_terms_matrix,
        length(all_terms),
	length(orders),
        orders,
        X,
    )
end


function with_coeffs(row,bern_terms,n)
	scales = [
	    iszero((i - 1) % n) ? row[(i - 1) ÷ n + 1] : one(eltype(bern_terms))
	    for i in axes(bern_terms, 1)
	    ]
	return scales .* bern_terms
end
function get_poly_low(poly_idx::Int, interval::CombinedPolyBernsteinInterval)
    #@show size(interval.Low)
    #@show size(interval.bern_terms)
    return with_coeffs(interval.Low[poly_idx,:],interval.bern_terms, interval.n)
end
function get_poly_up(poly_idx::Int, interval::CombinedPolyBernsteinInterval)
    return with_coeffs(interval.Up[poly_idx,:],interval.bern_terms, interval.n)
end
"""
Construct a CombinedBernsteinInterval over h by filling Low and Up with x for the unfixed variables
"""
function init_combined_bernstein_interval(h::Hyperrectangle)
    num_polys = dim(h)
    unfixed_mask = (h.radius .!= 0)
    n = count(unfixed_mask)
    X = Hyperrectangle(h.center[unfixed_mask], h.radius[unfixed_mask])
    #@show unfixed_mask,n
    @polyvar x[1:num_polys]
    bern_terms_matrix  = similar(h.radius, (n+1)*n,2)
    coeff_matrix  = zeros(eltype(h.radius), num_polys ,n+1  )
    #@show coeff_matrix
    
    # one term x_i for each unfixed variable
    for x_i in eachindex(X.radius)
	bern_terms_matrix_idx = (x_i-1)*n +1
	bern_terms_matrix[bern_terms_matrix_idx : bern_terms_matrix_idx + n - 1, :]= init_one_coeff_mat(x_i,n,x_i,X)
    end
    # one constant term (1) at the end.
    bern_terms_matrix[(n*n) + 1:(n+1)*n  , :] .=  1.0

    # set coefficient vectors 
    for poly_idx in eachindex(h.radius)
	if unfixed_mask[poly_idx]
	   x_index =  count(@view unfixed_mask[1:poly_idx]) 
	   coeff_matrix[poly_idx, x_index]  =1
	else 
	    coeff_matrix[poly_idx, n+1]  = h.center[poly_idx]
	end
    end

    return CombinedPolyBernsteinInterval(
	copy(coeff_matrix),coeff_matrix,bern_terms_matrix,n+1,n,fill(1,n),X
    )
end
function constant_term(type, n, orders)
    
    term = zeros(type,n,maximum(orders)+1)
    for i in axes(term,1)
	term[i,1:orders[i]+1] .= one(type)
    end
    return term

end
function translate(interval::CombinedPolyBernsteinInterval,b)
    return translate(interval,b,b)
end

function translate(interval::CombinedPolyBernsteinInterval,b_low, b_up) 
   #  look for constant term in the polynomial and increase coefficents at the index: 
    for i in 1:interval.t
	term = @view interval.bern_terms[(i-1)*interval.n + 1:i*interval.n,:] 
	if all(x -> x==1,term)
	    # constant term found
	    low = hcat(
		interval.Low[:,1:i-1],
		interval.Low[:,i]+b_low,
		interval.Low[:,i+1:end],
	    )
	    up = hcat(
		interval.Up[:,1:i-1],
		interval.Up[:,i]+b_up,
		interval.Up[:,i+1:end],
	    )
	    return CombinedPolyBernsteinInterval(low,up , copy(interval.bern_terms),interval.t,interval.n,copy(interval.orders),interval.X)
	end
    end
    # no constant term found therefore add it at the end of the matrix
    new_bterm = vcat(interval.bern_terms,constant_term(eltype(interval.bern_terms),interval.n,interval.orders))
    return CombinedPolyBernsteinInterval( [interval.Low b_low],[ interval.Up b_up], new_bterm,interval.t+1,interval.n,copy(interval.orders),interval.X)
	
end

"""
Interval map overapproximating an affine map Wx + b for x ∈ I 

W⁻ - (matrix) negative weights
W⁺ - (matrix) positive weights
I  - (BernsteinInterval) bernsteinInterval
b  - (vector) bias
"""
function interval_map(W⁻, W⁺, I::CombinedPolyBernsteinInterval, b; use_memory_optimizations=true)
    #@show W⁻ , W⁺
    new_low = W⁻ * I.Up + W⁺ * I.Low
    new_up = W⁻ * I.Low + W⁺ * I.Up
    return translate(CombinedPolyBernsteinInterval(new_low,new_up,I.bern_terms,I.t, I.n, I.orders, I.X),b)
end

function square(interval::CombinedPolyBernsteinInterval;use_memory_optimizations=true)::CombinedPolyBernsteinInterval 
    new_t = interval.t^2 
    new_orders = 2 .* interval.orders
    C = repeat(binomial_tensor(interval.orders),interval.t)
    C_rescale = repeat(binomial_tensor(new_orders),new_t)
    scaled = interval.bern_terms .* C 
    
    scaled_a = reduce(vcat, [
	repeat(scaled[(term_idx-1)*interval.n + 1 : term_idx*interval.n, :], interval.t)
	for term_idx in 1:interval.t
    ])

    scaled_b = repeat(scaled,interval.t)

    conv = row_convolution_kernel(scaled_a,scaled_b)

    new_terms = ifelse.(C_rescale .!= 0, conv ./ C_rescale, 0.0)

    n_rows, n_cols = size(interval.Low)

    # (n_rows, n_cols, 1) .* (n_rows, 1, n_cols) -> (n_rows, n_cols, n_cols), dann reshape
    low_outer = reshape(interval.Low, n_rows, n_cols, 1) .* reshape(interval.Low, n_rows, 1, n_cols)
    new_low = reshape(low_outer, n_rows, n_cols^2)
    
    up_outer = reshape(interval.Up, n_rows, n_cols, 1) .* reshape(interval.Up, n_rows, 1, n_cols)
    new_up = reshape(up_outer, n_rows, n_cols^2)

    #new_low = similar(interval.Low,size(interval.Low,1),new_t)
    #new_up = similar(interval.Up,size(interval.Up,1),new_t)
    #i =1 
    #for c1 in axes(interval.Low,2)
    #    for c2 in axes(interval.Low,2)
    #        @views new_low[:,i] .=  interval.Low[:,c1] .* interval.Low[:,c2]
    #        @views new_up[:,i] .=  interval.Up[:,c1] .* interval.Up[:,c2]
    #        i+=1
    #    end
    #end
    if use_memory_optimizations 
	return combine_terms(CombinedPolyBernsteinInterval(new_low,new_up,new_terms,new_t,interval.n,new_orders,interval.X))
    end
    return CombinedPolyBernsteinInterval(new_low,new_up,new_terms,new_t,interval.n,new_orders,interval.X)
end
function elevate_all_to(interval::CombinedPolyBernsteinInterval, new_orders::Vector{Int64};use_memory_optimizations=true)
    @assert all(sub -> all(sub .<= new_orders), interval.orders)
    #@show interval
    diffs = new_orders .- interval.orders
    bin_tensor= repeat(binomial_tensor(interval.orders),interval.t)

    res = interval.bern_terms .* bin_tensor


    C_diff = repeat(binomial_tensor(diffs), interval.t,1) 

    C_new_orders = repeat(binomial_tensor(new_orders), interval.t, 1)

    result = ifelse.(C_new_orders .!= 0, row_convolution_kernel(res,C_diff) ./ C_new_orders, 0.0)
    return CombinedPolyBernsteinInterval(copy(interval.Low), copy(interval.Up),result, interval.t,interval.n, copy(new_orders),interval.X)
end
function add(inter_a::CombinedPolyBernsteinInterval, inter_b::CombinedPolyBernsteinInterval;use_memory_optimizations=true) 
    new_order = max.(inter_a.orders, inter_b.orders)
    elevated_a = elevate_all_to(inter_a, new_order) 
    elevated_b = elevate_all_to(inter_b, new_order) 
   
    if use_memory_optimizations 
	return combine_terms(CombinedPolyBernsteinInterval( [elevated_a.Low elevated_b.Low], [elevated_a.Up elevated_b.Up],[elevated_a.bern_terms ; elevated_b.bern_terms], elevated_a.t + elevated_b.t, elevated_a.n, new_order, elevated_a.X ))
    end
	return CombinedPolyBernsteinInterval( [elevated_a.Low elevated_b.Low], [elevated_a.Up elevated_b.Up],[elevated_a.bern_terms ; elevated_b.bern_terms], elevated_a.t + elevated_b.t, elevated_a.n, new_order, elevated_a.X )
end


"""
Computes a one-dimensional quadratic map for each input dimension.
I.e. yᵢ =  qᵢxᵢ² for each input dimension i
"""
function quadratic_map_1d(qs_low, qs_up, interval::CombinedPolyBernsteinInterval; use_memory_optimizations=true)::CombinedPolyBernsteinInterval
    quad = square(interval;use_memory_optimizations)
    return CombinedPolyBernsteinInterval(quad.Low .* qs_low, quad.Up .* qs_up, quad.bern_terms, quad.t,quad.n,quad.orders,quad.X,quad.lbs,quad.ubs)
end


"""
Computes a one-dimensional quadratic function for each input dimension.
I.e. yᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ for each input dimension i
"""
function quadratic_propagation(a_low,  b_low, c_low,a_up, b_up, c_up, inter::CombinedPolyBernsteinInterval; use_memory_optimizations=true)

    p_quad = quadratic_map_1d(a_low,a_up, inter; use_memory_optimizations)
    lin_low_coeffs = inter.Low .* b_low
    lin_up_coeffs = inter.Up .* b_up

    sum =  add( CombinedPolyBernsteinInterval(lin_low_coeffs,lin_up_coeffs, inter.bern_terms,inter.t, inter.n, inter.orders,inter.X) , p_quad; use_memory_optimizations)
    
    return translate(sum,c_low,c_up)
    
end
function unique_idx(term,orders,j)
    #n = length(orders)
    #left_prod = one(eltype(term))
    #right_prod = one(eltype(term))
    #for row_idx in axes(term,1)
    #    left_prod *= term[row_idx,1]
    #    right_prod *= term[row_idx,orders[row_idx]]
    #end
    if all(term[1,:] .<= 0.0)
        return orders[j] + 1,1
    end 
        return 1, orders[j] + 1
end

"""
Max/Min eines einzelnen Terms (implizite Form), ohne den vollen Tensor zu bilden.
bvecs[m] ist monoton (steigend oder fallend); a ist der Term-Koeffizient.
"""
function term_extrema(term, orders)
    n = length(orders)
    left_prod = one(eltype(term))
    right_prod = one(eltype(term))
    for row_idx in axes(term,1)
        left_prod *= term[row_idx,1]
        right_prod *= term[row_idx,orders[row_idx]+1]
    end
    return min(left_prod, right_prod), max(left_prod, right_prod)
end
function term_extremas( term,orders)
    lo = 1
    hi = 1
    for i in axes(term,1)
        left, right = term[i,1], term[i,orders[i]+1]   # bei monotonen Vektoren: O(1) via bv[1],bv[end], hier der Einfachheit halber extrema()

        candidates = (lo*left, lo*right, hi*left, hi*right)
        lo, hi =  minimum(candidates), maximum(candidates)
    end
    return lo, hi
end
function term_min_step_along_j(term,orders, j)
    d = diff(term[j,1:orders[j]+1])                 # Differenzen entlang Dimension j (1D, billig)
    min_step_j = minimum(abs.(d))      # da bvecs[j] monoton: d hat konsistentes Vorzeichen
    other_factor = one(eltype(term)) 
    for m  in axes(term,1)
        m == j && continue
	#@assert min(term[m,1],term[m,orders[m]+1]) == term[m,1]
        other_factor *= min(term[m,1], term[m,orders[m]+1])   # bei monotonen bv: min(bv[1],bv[end])
    end
    return min_step_j * abs(other_factor)
end
"""
Schätzt den 'Steilheits'-Beitrag eines Terms zur Dominance-Summe entlang Dimension j,
nur anhand der beiden Randdifferenzen (0↔1 und deg-1↔deg) statt des vollen Vektors.
Nutzt aus, dass die konsekutiven Differenzen der Bernstein-Koeffizienten selbst monoton
sind, sodass Min/Max der Differenzen garantiert an einem der beiden Ränder liegen.
"""
function term_diff_bound( term, orders, j)
    l_j = orders[j] + 1

    # Randdifferenzen entlang j (beide Enden testen, breitere nehmen)
    left_diff  = abs(term[j,2] - term[j,1])
    right_diff = abs(term[j,l_j] - term[j,l_j-1])
    step_j = max(left_diff, right_diff)

    val = step_j * one(eltype(term))
    for i in axes(term,1)
        i == j && continue
        lo, hi = term[i,1], term[i,orders[i]+1]
        val *= max(abs(lo), abs(hi))   # größte Magnitude der übrigen Faktoren
    end
    return val
end

term_width(term,orders) = ((lo,hi) = term_extrema(term, orders); hi - lo)
term_width2(term,orders) = ((lo,hi) = term_extremas(term, orders); hi - lo)
function bound_algo(as,orders::Vector{Int},t::Int,j::Int,constant_terms,term_widths,min_steps_j)
    n = length(orders)
    non_constant_indices = Int[]
    sizehint!(non_constant_indices, t)
    l_j = orders[j]+1
    #for t_idx in 1:t
    #    # all 1. is x^0 -> term is constant with respect to x_j || all 0. is O*x 
    #    
	#if #isapprox(as[t_idx], 0) || constant_terms[t_idx,j]
    #        continue
    #    end 
    #    push!(non_constant_indices,t_idx)
    #end
    non_constant_mask = .!( isapprox.(as,0) .|| constant_terms[:,j])
    # exactly one term is non-constant with respect to x_j
    #if length(non_constant_indices) == 1
    #    min_idx,max_idx =  as[non_constant_indices[1]] > 0.0 ? (1, l_j ) : (l_j, 1)
    #    return min_idx:min_idx, max_idx:max_idx
    #end
    if count(non_constant_mask) == 1
        min_idx,max_idx =  as[non_constant_mask][1] > 0.0 ? (1, l_j ) : (l_j, 1)
        return min_idx:min_idx, max_idx:max_idx
    end
    # more than one non constant term therfore continue with monotonicity_test
    #increasing_mask = as[non_constant_mask] .>= 0
    increasing_mask = non_constant_mask .&& as .> 0
    decreasing_mask = non_constant_mask .&& as .< 0


    if all(!,increasing_mask)
        return l_j:l_j, 1:1
    end
    if all(!,decreasing_mask)
        return 1:1, l_j:l_j
    end
    
    #width_dec = sum(term_widths[i]* abs(as[i]) for i in decreasing_indices)
    width_dec = sum( term_widths[decreasing_mask] .* abs.(as[decreasing_mask]) )


    #diff_inc = sum(min_steps_j[i][j]*as[i]  for i in increasing_indices)
    diff_inc = sum(min_steps_j[increasing_mask,j] .* as[increasing_mask]  )
    
    @assert diff_inc >= 0  "$diff_inc , $(as[increasing_indices]),"

    if diff_inc > width_dec
        return   1:1,l_j:l_j
    end
    

    width_inc = sum( term_widths[increasing_mask] .* abs.(as[increasing_mask]) )
    #width_inc = sum(term_widths[i]* abs(as[i]) for i in increasing_indices)
    #diff_dec = sum(min_steps_j[i][j]*abs(as[i])  for i in decreasing_indices)
    diff_dec = sum(min_steps_j[decreasing_mask,j] .* abs.(as[decreasing_mask])  )
    @assert diff_dec >= 0  "$diff_dec , $(as[decreasing_mask]) "
    if diff_dec > width_inc
        return   l_j:l_j,1:1
    end

    return 1:l_j, 1:l_j
end
"""
Berechnet den Tensor aller Bernstein-Koeffizienten im Suchbereich S = (S[1],...,S[n]),
indem für jeden Term das Outer-Product der (auf S eingeschränkten) univariaten Faktoren
gebildet und über alle Terme aufsummiert wird — ohne pro Index einzeln zu iterieren.
"""
#function evaluate_reduced_tensor(bern_mat,as, t,n,  S)
#    out = nothing
#    for t_idx in 1:t
#        rows = get_term(t_idx, n)
#        a = as[t_idx]
#        iszero(a) && continue
#        factors = ntuple(n) do m
#            sub = @view bern_mat[rows[m], S[m]]
#            shape = ntuple(d -> d == m ? length(S[m]) : 1, n)
#            reshape(sub, shape...)
#        end
#        term = reduce(.*, factors)
#        if out === nothing
#            out = a .* term          # Koeffizient hier, nicht via Matrixkopie
#        else
#            out .+= a .* term
#        end
#    end
#    return out
#end
global calls_0d=0
global calls_1d=0
global calls_2d=0
global calls_higher=0
function _eval_broadcast(bern_mat, as::Vector{TA}, t, n, S, nonscalar, dims::NTuple{K,Int}) where {K,TA}
    T = promote_type(eltype(bern_mat), TA)
        # --- Sonderfall K == 0: reines Skalarprodukt ---
    if K == 0
        s = zero(T)
        for t_idx in 1:t
            a = as[t_idx]
            iszero(a) && continue
            rows = get_term(t_idx, n)
            p = a
            @inbounds for m in 1:n
                p *= bern_mat[rows[m], first(S[m])]
            end
            s += p
        end
        return (s, s)          # min == max, ein einzelner Punkt
    end
    @show dims
    out = zeros(T, dims)
    for t_idx in 1:t
        a = as[t_idx]; iszero(a) && continue
        rows = get_term(t_idx, n)
        coeff = a
        @inbounds for m in 1:n
            length(S[m]) == 1 && (coeff *= bern_mat[rows[m], first(S[m])])
        end
        iszero(coeff) && continue
        factors = ntuple(K) do k          # K compile-time → NTuple{K}, stabil
            mk = nonscalar[k]
            sub = @view bern_mat[rows[mk], S[mk]]
            shape = ntuple(d -> d == k ? dims[k] : 1, K)
            reshape(sub, shape...)
        end
        out .+= coeff .* .*(factors...)
    end
    return extrema(out)
end
function evaluate_reduced_tensor(bern_mat, as, t, n, S)
    lens = ntuple(m -> length(S[m]), n)
    nonscalar = [m for m in 1:n if lens[m] > 1]   # nur die "echten" Dimensionen
    length(nonscalar) == 0 && (global calls_0d +=1)
    length(nonscalar) == 1 && (global calls_1d +=1)
    length(nonscalar) == 2 && (global calls_2d +=1)
    length(nonscalar) > 2 && (global calls_higher +=1)
    #@show calls_0d,calls_1d,calls_2d,calls_higher
    dims = ntuple(k -> lens[nonscalar[k]], length(nonscalar))
    return _eval_broadcast(bern_mat,as,t,n,S,nonscalar,dims)

end
#function evaluate_reduced_tensor(bern_mat, as, t, n, S)
#     
#    dims = ntuple(m -> length(S[m]), n)
#    T = promote_type(eltype(bern_mat), eltype(as))
#    out = zeros(T, dims)
#    for t_idx in 1:t
#        a = as[t_idx]
#        iszero(a) && continue
#        rows = get_term(t_idx, n)
#        factors = ntuple(n) do m
#            sub = @view bern_mat[rows[m], S[m]]
#            shape = ntuple(d -> d == m ? length(S[m]) : 1, n)
#            reshape(sub, shape...)
#        end
#        # out += a * (f1 ⊗ f2 ⊗ ... ⊗ fn), alles in einem fusionierten Loop
#        out .+= a .* .*(factors...)
#    end
#    return extrema(out)
#end
function faster_exact_bounds(as,bern_mat::AbstractArray,orders::AbstractArray,t::Int,constant_terms, term_widths, min_steps_j)
    n = length(orders)
    S_max =   Vector{UnitRange{Int64}}(undef, n)
    S_min = Vector{UnitRange{Int64}}(undef, n)
    # expand bern_mat 

    for x_i in 1:n
	S_min[x_i],S_max[x_i] = bound_algo(as,orders,t,x_i,constant_terms,term_widths,min_steps_j)
    end
    threshold = 500_000
    min_possibilites = prod(length, S_min)
    if min_possibilites > threshold || min_possibilites < 0 # check for overflow
	    println(" $min_possibilites is too much using shortcut")

        nvars = length(orders)
        b_min = zero(eltype(bern_mat))
        b_max = zero(eltype(bern_mat))

        for term in 0:(t-1)
            left_prod  = as[term+1]
            right_prod = as[term+1]

            for var in 1:nvars
                coeffs = @view bern_mat[term*nvars  + var, :]

                first_idx = 1  #findfirst(!iszero, coeffs)
                last_idx  = orders[var] +1  #findlast(!iszero, coeffs)

                left_prod *= coeffs[first_idx]
                right_prod *= coeffs[last_idx]
            end
            b_min += min(left_prod,right_prod)
            b_max += max(left_prod,right_prod)
        end
    elseif S_min == S_max 
        b_min, b_max = evaluate_reduced_tensor(bern_mat,as,t,n,S_min)
    else
        b_min,_ = evaluate_reduced_tensor(bern_mat,as,t,n,S_min)
        _, b_max = evaluate_reduced_tensor(bern_mat,as,t,n,S_max)
    end
    #for idx in  CartesianIndices(Tuple(S_min))
    #    b =  dense(bern_mat,t,orders,idx)
    #    if b < b_min 
    #        b_min = b 
    #    end
    #end
    #b_max = -Inf
    #for idx in  CartesianIndices(Tuple(S_max))
    #    b =  dense(bern_mat,t,orders,idx)
    #    if b > b_max
    #        b_max = b 
    #    end
    #end
    return b_min, b_max 
end

function precompute(bern_mat, t, orders) 
    n = length(orders)
    constant_terms = BitArray(undef,t, length(orders))
    for t_idx in 1:t 
	term = @view bern_mat[get_term(t_idx,n),:]
	for var in 1:n 
	    row = @view term[var,1:orders[var]+1]
	    constant_terms[t_idx,var] = all(x -> isapprox.(x,1.0), row) ||   all(x -> isapprox(x,0.0),row)
	end

    end
    term_widths = [term_width2((@view bern_mat[get_term(t_idx,n),:]) ,orders) for t_idx in 1:t ]
    min_steps_j = [term_min_step_along_j((@view bern_mat[get_term(t_idx, n), :]), orders, j)
               for t_idx in 1:t, j in 1:n]
    return constant_terms,term_widths, min_steps_j
end

function bounds(interval::CombinedPolyBernsteinInterval;  use_shortcut=true)
    num_p = size(interval.Low,1)
    T = eltype(interval.Low)
    llbs = Vector{T}(undef,num_p)
    lubs = Vector{T}(undef,num_p)
    ulbs = Vector{T}(undef,num_p)
    uubs = Vector{T}(undef,num_p)
    TS = Vector{UnitRange{Int64}}
    lS_mins = Vector{TS}(undef,num_p)
    lS_maxs = Vector{TS}(undef,num_p)
    uS_mins = Vector{TS}(undef,num_p)
    uS_maxs = Vector{TS}(undef,num_p)

    #llbs = eltype(interval.Low)[]# similar(interval.Low,1)#, size(interval.Low,1))
    #lubs = eltype(interval.Low)[]# similar(interval.Low,1)#, size(interval.Low,1))
    #ulbs = eltype(interval.Up)[]#similar(interval.Up,1)#, size(interval.Up,1))
    #uubs = eltype(interval.Up)[]#similar(interval.Up,1)#, size(interval.Up,1))
    
    #order_first_var = interval.orders[1]
    #monomon_coefficients = [ calculate_bern_coeff_for_monomial(e,order_first_var,low(interval.X)[1],high(interval.X)[1]) for e in 0:order_first_var ]
    #inverse = inv(stack(monomon_coefficients))
    #@threads for p_idx in axes(interval.Low,1)
    #dense_repr_terms = dense_new(interval.bern_terms,interval.t,interval.orders)
    #reshape_shape = ntuple(_ -> 1, ndims(dense_repr_terms)-1)...,interval.t
    if use_shortcut 
	@assert false
	for p_idx in axes(interval.Low,1)
	    low_poly = get_poly_low(p_idx,interval)
	    up_poly = get_poly_up(p_idx,interval)
	    llbsi , lubsi  = quadrant_ibf_minmax(low_poly,interval.t,interval.orders)#,monomon_coefficients, inv(stack(monomon_coefficients)))
	    ulbsi, uubsi = quadrant_ibf_minmax(up_poly,interval.t,interval.orders)#,monomon_coefficients, inv(stack(monomon_coefficients)))

	    llbs[p_idx] = llbsi  
	    lubs[p_idx] = lubsi  
	    ulbs[p_idx] = ulbsi  
	    uubs[p_idx] = uubsi  
	end

	return llbs,lubs, ulbs, uubs
    end

    t_start = time()
    c0d = calls_0d
    c1d = calls_1d
    c2d = calls_2d
    chd = calls_higher
    constant_terms,term_widths , min_steps_j = precompute(interval.bern_terms, interval.t,interval.orders)
     

    #terms_to_add = []
    #for t_idx in 1:interval.t 
    #    first_row = @view interval.bern_terms[(t_idx-1)*interval.n + 1,1:interval.orders[1]+1 ]
    #    # if multiple of monomon_coefficient normal case otherwise expand into combined terms
    #        if any(are_multiples(first_row , monomon_coefficient) for monomon_coefficient in monomon_coefficients)
    #        continue
    #    end
    #    original_coeffs = inverse * first_row
    #        if all(original_coeffs .== 0 ) 
    #    	    continue 
    #        end
    #        for row in  monomon_coefficients .* original_coeffs
    #        term = [row... zero(size(interval.bern_terms,2)-length(row)); interval.bern_terms[(t_idx-1) *interval.n+2: t_idx*interval.n,:] ]
    #        push!(terms_to_add, term)
    #        end

    #end
    #@assert length(terms_to_add) == 0 "removing the term that is replaced must still be done"
    #bern_mat = vcat(interval.bern_terms, terms_to_add...)
    #t += length(terms_to_add)

    # i is row idx for:  [Low; Up]
    unique_as  =  Vector{Int64}()
    low_evaluated_poly = Array{Int64}(undef,size(interval.Low,1))
    for i  in axes(interval.Low,1)
        idx = findfirst( x -> isapprox((@view interval.Low[i,:]), (@view interval.Low[x,:]) ) , unique_as)
        if isnothing(idx)
            push!(unique_as,i)
            low_evaluated_poly[i] =size(unique_as,1)
        else 
            low_evaluated_poly[i] = idx
        end
    end
    up_evaluated_poly =  Array{Int64}(undef,size(interval.Up,1))
    for i  in axes(interval.Up,1)
        idx = findfirst( x -> isapprox((@view interval.Up[i,:]),(@view interval.Up[x,:])) , unique_as)
        if isnothing(idx)
            push!(unqiue_as,i + num_p)
            up_evaluated_poly[i] =size(unique_as,1)
        else 
            up_evaluated_poly[i] = idx
        end
    end
    println("saved: $(2*size(interval.Low,1) - size(unique_as,1) )")

    unique_bounds = Array{Tuple{Float64,Float64}}(undef,size(unique_as,1))
    for i in eachindex(unique_as)
        p_idx = unique_as[i]
        if p_idx > num_p
            unique_bounds[i] = faster_exact_bounds(interval.Up[p_idx-num_p,:] ,interval.bern_terms,interval.orders,interval.t,constant_terms,term_widths,min_steps_j) 
        else
            unique_bounds[i] = faster_exact_bounds(interval.Low[p_idx,:] ,interval.bern_terms,interval.orders,interval.t,constant_terms,term_widths,min_steps_j) 
        end
    end



    for p_idx in axes(interval.Low,1)

        #llbsi, lubsi ,lS_min,lS_max= faster_exact_bounds(interval.Low[p_idx,:] ,interval.bern_terms,interval.orders,interval.t,constant_terms, term_widths, min_steps_j)
            #up_dense = dense_new(up_poly,interval.t,interval.orders)
            #ulbsi = minimum(up_dense)
            #uubsi = maximum(up_dense)
            #ulbsi,uubsi = dense_min_max(up_poly,interval.t,interval.orders)
        #ulbsi, uubsi, uS_min, uS_max = faster_exact_bounds(interval.Up[p_idx,:],interval.bern_terms,interval.orders,interval.t,constant_terms, term_widths,min_steps_j)
            
    
        llbs[p_idx], lubs[p_idx] = unique_bounds[low_evaluated_poly[p_idx]]
        ulbs[p_idx], uubs[p_idx] = unique_bounds[up_evaluated_poly[p_idx]]
	   # llbs[p_idx] = llbsi  
	   # lubs[p_idx] = lubsi  
	   # ulbs[p_idx] = ulbsi  
	   # uubs[p_idx] = uubsi  

    end
    @show time()- t_start
    c0dn = (calls_0d) - c0d
    c1dn = (calls_1d) - c1d
    c2dn = (calls_2d) - c2d
    chdn = (calls_higher) - chd
    @show c0dn, c1dn, c2dn,chdn
    #@show lS_maxs, lS_mins, uS_maxs, uS_mins
    return llbs,lubs,ulbs,uubs

end

"""
Calculates concrete bounds for A*s + b for BernsteinPoly s with common generators.
"""
function bounds(A::AbstractMatrix, b::AbstractVector, s::CombinedPolyBernsteinInterval; use_shortcut=true)
    mapped_interval = interval_map(
        min.(0, A),
        max.(0, A),
        s,
        b,
    )
    ll, _ ,_, uu = bounds(mapped_interval; use_shortcut)
    return ll, uu
end

function get_term(t_idx, n) 
    return (t_idx-1) * n +1 : t_idx *n
end
"""
remove duplicate terms
"""
function combine_terms(inter::CombinedPolyBernsteinInterval)
   unique_terms = Matrix{eltype(inter.bern_terms)}[]
   Low =  zeros(eltype(inter.Low), size(inter.Up,1), 0)   # num_polys×0 matrix
   Up =  zeros(eltype(inter.Up), size(inter.Up,1), 0)   # num_polys×0 matrix
   for old_term_idx in 1:inter.t 
	term = inter.bern_terms[ get_term(old_term_idx,inter.n), : ]
	t_idx = findfirst(x -> x == term,unique_terms)
	if isnothing(t_idx)
	    unique_terms = [unique_terms...,copy(term)]
	    Low = [Low zeros(eltype(Low),size(Low,1))]
	    Up = [Up zeros(eltype(Up),size(Up,1))]
	    t_idx = length(unique_terms)
	end
	#Low[:,t_idx] .+=  inter.Low[:, old_term_idx]
	Low = hcat(Low[:,1:t_idx-1], 
		    Low[:,t_idx] .+ inter.Low[:,old_term_idx],
		    Low[:,t_idx+1:end])
	Up = hcat(Up[:,1:t_idx-1], 
		    Up[:,t_idx] .+ inter.Up[:,old_term_idx],
		    Up[:,t_idx+1:end])
	#Up[:,t_idx] .+=  inter.Up[:, old_term_idx]
    end
    # look for zero columns: 
    new_Low =  zeros(eltype(inter.Low), size(inter.Low,1), 0)   # num_polys×0 matrix
    new_Up =  zeros(eltype(inter.Up), size(inter.Up,1), 0)   # num_polys×0 matrix
    new_unique_terms  = Matrix{eltype(inter.bern_terms)}[]
    for col_idx in axes(Low,2)
	low_col = @view Low[:,col_idx]
	up_col = @view Up[:,col_idx]
	if all(low_col .== 0) && all(up_col .== 0)
	    continue
	end
	new_unique_terms = [new_unique_terms... ,unique_terms[col_idx]]
	new_Low = [new_Low low_col]
	new_Up = [new_Up up_col]
	
    end
    if isempty(new_unique_terms)
	new_unique_terms = [new_unique_terms..., constant_term(eltype(inter.bern_terms),inter.n,inter.orders)]
	new_Low =  zeros(eltype(inter.Low), size(inter.Up,1), 1)   # num_polys×1 matrix
	new_Up =  zeros(eltype(inter.Up), size(inter.Up,1), 1)   # num_polys×1 matrix
    end
    return CombinedPolyBernsteinInterval(new_Low,new_Up,vcat(new_unique_terms...),length(new_unique_terms),inter.n,inter.orders,inter.X,inter.lbs,inter.ubs)

end
