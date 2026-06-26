using LazySets
using DynamicPolynomials

struct CombinedMultiBernsteinImp{N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    coeffs::TN # each row of the matrix corresponds to the term coefficients for one Low polynomial
    bern_terms::TN # matrix containing the bernstein terms without coefficients. 
    t::M # number of terms
    n::M # input dimension
    orders::Vector{Int64} # orders of the variables
    X::Hyperrectangle{N}
end

struct CombinedPolyBernsteinInterval{N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}, VN<:AbstractArray{N}}
    Low::CombinedMultiBernsteinImp{N,M,O,TN}
    Up::CombinedMultiBernsteinImp{N,M,O,TN}
    lbs::Vector{VN}  # (vector of vectors) lower bounds for intermediate values
    ubs::Vector{VN}  # (vector of vectors) upper bounds for intermediate values
end

function CombinedPolyBernsteinInterval(Low::CombinedMultiBernsteinImp{N,M,O,TN}, Up::CombinedMultiBernsteinImp{N,M,O,TN}) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    return CombinedPolyBernsteinInterval(Low, Up,Vector{Vector{N}}(), Vector{Vector{N}}())
end


function with_coeffs(row,bern_terms,n)
	scales = ones(eltype(row),  size(bern_terms,1))
	scales[1:n:end] .= row
	return scales .* bern_terms
end
function get_poly(poly_idx::Int, multi::CombinedMultiBernsteinImp)
    return with_coeffs(multi.coeffs[poly_idx,:],multi.bern_terms, multi.n)
end
"""
Construct a CombinedBernsteinInterval over h by filling Low and Up with x for the unfixed variables
"""
function init_combined_bernstein_interval(h::Hyperrectangle)
    num_polys = dim(h)
    unfixed_mask = (h.radius .!= 0)
    n = count(unfixed_mask)
    X = Hyperrectangle(h.center[unfixed_mask], h.radius[unfixed_mask])
    @show X
    #@show unfixed_mask,n
    @polyvar x[1:num_polys]
    bern_terms_matrix  = similar(h.radius, (n+1)*n,2)
    coeff_matrix  = zeros(eltype(h.radius), num_polys ,n+1  )
    @show coeff_matrix
    
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
	CombinedMultiBernsteinImp(coeff_matrix,bern_terms_matrix,n+1,n,fill(1,n),X),
	CombinedMultiBernsteinImp(copy(coeff_matrix),copy(bern_terms_matrix),n+1,n,copy(fill(1,n)),copy(X)),
    )
end

function translate(multi::CombinedMultiBernsteinImp,b) 
   #  look for constant term in the polynomial and increase coefficents at the index: 
    for i in 1:multi.t
	term = @view multi.bern_terms[(i-1)*multi.n + 1:i*multi.n,:] 
	if all(x -> x==1,term)
	    # constant term found
	    coeffs = copy(multi.coeffs)
	    coeffs[:,i ] .+= b 
	    return CombinedMultiBernsteinImp(coeffs , copy(multi.bern_terms),multi.t,multi.n,copy(multi.orders),multi.X)
	end
    end
    # no constant term found therefore add it at the end of the matrix
    constant_term = ones(eltype(multi.bern_terms),multi.n,size(multi.bern_terms,2))
    new_bterm = vcat(multi.bern_terms,constant_term)
    return CombinedMultiBernsteinImp([ multi.coeffs b], new_bterm,multi.t+1,multi.n,copy(multi.orders),multi.X)
	
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
    @assert I.Up.orders == I.Low.orders
    new_low = W⁻ * I.Up.coeffs + W⁺ * I.Low.coeffs
    new_up = W⁻ * I.Low.coeffs + W⁺ * I.Up.coeffs
    return CombinedPolyBernsteinInterval(
	translate(CombinedMultiBernsteinImp(new_low,I.Low.bern_terms,I.Low.t, I.Low.n, I.Low.orders, I.Low.X),b),
	translate(CombinedMultiBernsteinImp(new_up,I.Up.bern_terms,I.Up.t, I.Up.n, I.Up.orders, I.Up.X),b)
	,I.lbs,I.ubs);
end

function square(multi::CombinedMultiBernsteinImp;use_memory_optimizations=true)::CombinedMultiBernsteinImp 
    new_t = multi.t^2 
    new_orders = 2 .* multi.orders
    C = repeat(binomial_tensor(multi.orders),multi.t)
    C_rescale = repeat(binomial_tensor(new_orders),new_t)
    scaled = multi.bern_terms .* C 
    
    scaled_a = similar(scaled, multi.t^2 * multi.n, size(scaled,2))
    # scaled_a contains every term t times
    for term_idx in 1:multi.t 
	scaled_a[(term_idx-1)* multi.n * multi.t + 1 : term_idx * multi.n * multi.t   , :]  = repeat(scaled[(term_idx-1)*multi.n + 1: term_idx * multi.n, :],multi.t)
    end

    scaled_b = repeat(scaled,multi.t)

    conv = row_convolution_kernel(scaled_a,scaled_b)
    @show C_rescale
    @show conv
    new_terms = ifelse.(C_rescale .!= 0, conv ./ C_rescale, 0.0)

    new_c = similar(multi.coeffs,size(multi.coeffs,1),new_t)
    i =1 
    for c1 in axes(multi.coeffs,2)
	for c2 in axes(multi.coeffs,2)
	    @show c1,c2
	    @views new_c[:,i] .=  multi.coeffs[:,c1] .* multi.coeffs[:,c2]
	    i+=1
	end
    end
    return CombinedMultiBernsteinImp(new_c,new_terms,new_t,multi.n,new_orders,multi.X)
end
function elevate_all_to(multi::CombinedMultiBernsteinImp, new_orders::Vector{Int64};use_memory_optimizations=true)
    @assert all(sub -> all(sub .<= new_orders), multi.orders)
    diffs = new_orders .- multi.orders
    bin_tensor= repeat(binomial_tensor(multi.orders),multi.t)

    res = multi.bern_terms .* bin_tensor


    C_diff = repeat(binomial_tensor(diffs), multi.t,1) 

    C_new_orders = repeat(binomial_tensor(new_orders), multi.t, 1)

    result = ifelse.(C_new_orders .!= 0, row_convolution_kernel(res,C_diff) ./ C_new_orders, 0.0)
    return CombinedMultiBernsteinImp(copy(multi.coeffs),result, multi.t,multi.n, copy(new_orders),multi.X)
end
function add(multi_a::CombinedMultiBernsteinImp, multi_b::CombinedMultiBernsteinImp;use_memory_optimizations=true) 
    new_order = max.(multi_a.orders, multi_b.orders)
    elevated_a = elevate_all_to(multi_a, new_order) 
    elevated_b = elevate_all_to(multi_b, new_order) 
    
    return CombinedMultiBernsteinImp( [elevated_a.coeffs elevated_b.coeffs], [elevated_a.bern_terms ; elevated_b.bern_terms], elevated_a.t + elevated_b.t, elevated_a.n, new_order, elevated_a.X )
end


"""
Computes a one-dimensional quadratic map for each input dimension.
I.e. yᵢ =  qᵢxᵢ² for each input dimension i
"""
function quadratic_map_1d(qs, multi::CombinedMultiBernsteinImp; use_memory_optimizations=true)::CombinedMultiBernsteinImp
    quad = square(multi;use_memory_optimizations)
    quad.coeffs .= quad.coeffs .* qs
    return quad
end


"""
Computes a one-dimensional quadratic function for each input dimension.
I.e. yᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ for each input dimension i
"""
function quadratic_propagation(a, b, c, multi::CombinedMultiBernsteinImp; use_memory_optimizations=true)

    p_quad = quadratic_map_1d(a, multi; use_memory_optimizations)
    lin_coeffs = multi.coeffs .* b

    sum =  add( CombinedMultiBernsteinImp(lin_coeffs, multi.bern_terms,multi.t, multi.n, multi.orders,multi.X) , p_quad; use_memory_optimizations)
    
    return translate(sum,c)
    
end

function bounds(multi::CombinedMultiBernsteinImp, X::Hyperrectangle; use_shortcut=true)
    lbs = similar(multi.coeffs, size(multi.coeffs,1))
    ubs = similar(multi.coeffs, size(multi.coeffs,1))
    
    order_first_var = multi.orders[1]
    @show order_first_var
    monomon_coefficients = [ calculate_bern_coeff_for_monomial(e,order_first_var,low(multi.X)[1],high(multi.X)[1]) for e in 0:order_first_var ]
    for p_idx in axes(multi.coeffs,1)
	poly = get_poly(p_idx,multi)
	@show poly
	if use_shortcut
	    lbs[p_idx], ubs[p_idx] = quadrant_ibf_minmax(poly,multi.t,multi.orders,monomon_coefficients, inv(stack(monomon_coefficients)))
	else
	    lbs[p_idx], ubs[p_idx]= dense_min_max(poly,multi.t,multi.orders)
	end
    end
    return lbs,ubs

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
    ll, lu = bounds(mapped_interval.Low, mapped_interval.Low.X; use_shortcut)
    ul, uu = bounds(mapped_interval.Up, mapped_interval.Up.X; use_shortcut)
    return ll, uu
end
