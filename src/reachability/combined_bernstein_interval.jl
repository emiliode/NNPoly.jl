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

function bounds(interval::CombinedPolyBernsteinInterval;  use_shortcut=true)
    llbs = eltype(interval.Low)[]# similar(interval.Low,1)#, size(interval.Low,1))
    lubs = eltype(interval.Low)[]# similar(interval.Low,1)#, size(interval.Low,1))
    ulbs = eltype(interval.Up)[]#similar(interval.Up,1)#, size(interval.Up,1))
    uubs = eltype(interval.Up)[]#similar(interval.Up,1)#, size(interval.Up,1))
    
    order_first_var = interval.orders[1]
    monomon_coefficients = [ calculate_bern_coeff_for_monomial(e,order_first_var,low(interval.X)[1],high(interval.X)[1]) for e in 0:order_first_var ]
    #@threads for p_idx in axes(interval.Low,1)
    for p_idx in axes(interval.Low,1)
	low_poly = get_poly_low(p_idx,interval)
	up_poly = get_poly_up(p_idx,interval)
	if use_shortcut
	    llbsi , lubsi  = quadrant_ibf_minmax(low_poly,interval.t,interval.orders,monomon_coefficients, inv(stack(monomon_coefficients)))
	    llbs = [llbs..., llbsi]
	    lubs = [lubs..., lubsi]
	    ulbsi, uubsi = quadrant_ibf_minmax(up_poly,interval.t,interval.orders,monomon_coefficients, inv(stack(monomon_coefficients)))
	    ulbs = [ulbs..., ulbsi]
	    uubs = [uubs..., uubsi]

	else
	    #llbs[p_idx], lubs[p_idx] = dense_min_max_threaded(low_poly,interval.t,interval.orders)
	    #ulbs[p_idx], uubs[p_idx]= dense_min_max_threaded(up_poly,interval.t,interval.orders)
	    #llbs[p_idx], lubs[p_idx] = dense_min_max(low_poly,interval.t,interval.orders)
	    #ulbs[p_idx], uubs[p_idx]= dense_min_max(up_poly,interval.t,interval.orders)
	end
    end
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
