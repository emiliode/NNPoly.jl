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

# fix coefficient table for x^e = ((x'+1)/2)^e, e ∈ {0,1,2} 
const COEFFS2 = (
    (1.0,),            # e = 0: (x+1)/2^0 = 1*x^0
    (0.5, 0.5),        # e = 1: (x+1)/2 = 1/2 x^0 + 1/2x^1 
    (0.25, 0.5, 0.25), # e = 2: (x/2 + 1/2)^2 = x^2/4 + 2* x/2 * 1/2 + 1/4 = 1/4*x^0 + 1/2*x^1 + 1/4*x^2
)
function map_sp_to_correct_interval(E::AbstractArray{T}, G_Low::AbstractArray{T}, G_Up::AbstractArray) where{T} 
    p, h = size(E)
    n = size(G_Low, 1)
    @assert size(G_Low, 2) == h "E and G_Low need the same amount of columns"
    @assert size(G_Up, 2) == h "E and G_Up need the same amount of columns"
    @assert all(x -> 0 <= x <= 2, E) "this function requires degree <= 2 per variable."
 
    acc_low = Dict{NTuple{p,Int8}, Vector{T}}()
    acc_up = Dict{NTuple{p,Int8}, Vector{T}}()
    sizehint!(acc_low, min(h * 4, 3^min(p, 20)))   
    sizehint!(acc_up, min(h * 4, 3^min(p, 20)))   
 
    @inbounds for i in 1:h
        e = ntuple(k -> E[k, i], p)
        cvecs = ntuple(k -> COEFFS2[e[k] + 1], p)
        gcol_low  = @view G_Low[:, i]
        gcol_up  = @view G_Up[:, i]
 
        for jt in Iterators.product(ntuple(k -> 0:e[k], p)...)
            coeff = 1.0
            for k in 1:p
                coeff *= cvecs[k][jt[k] + 1]
            end
            key = NTuple{p,Int8}(jt)
 

            v_low = get(acc_low, key, nothing)
            if v_low === nothing
                acc_low[key] = coeff .* collect(gcol_low)
                acc_up[key] = coeff .* collect(gcol_up)
            else
                v_up = get(acc_up,key,nothing)
                v_low .+= coeff .* gcol_low
                v_up .+= coeff .* gcol_up
            end
        end
    end
 
    hnew = length(acc_low)
    Enew = Matrix{Int}(undef, p, hnew)
    Gnew_Low = Matrix{T}(undef, n, hnew)
    Gnew_Up = Matrix{T}(undef, n, hnew)
    for (col, (j, g)) in enumerate(acc_low)
        Enew[:, col] .= j
        Gnew_Low[:, col] .= g
    end
    for (col, (j,g)) in enumerate(acc_up)
        Gnew_Up[:,col] .= g
    end

    return  Enew, Gnew_Low, Gnew_Up
end

function ChainRulesCore.rrule(::typeof(map_sp_to_correct_interval),
                             E::AbstractArray{T}, G_Low::AbstractArray{T},
                             G_Up::AbstractArray) where {T}
    p, h = size(E)
    n = size(G_Low, 1)

    # --- Forward pass, while recording the scatter structure ---

    acc_low = Dict{NTuple{p,Int8}, Vector{T}}()
    acc_up  = Dict{NTuple{p,Int8}, Vector{T}}()
    key_to_col = Dict{NTuple{p,Int8}, Int}()   

    # contributions[i] = Vector of (out_col::Int, coeff::Float64)
    contributions = [Tuple{Int,Float64}[] for _ in 1:h]

    ncols = 0
    @inbounds for i in 1:h
        e = ntuple(k -> E[k, i], p)
        cvecs = ntuple(k -> COEFFS2[e[k] + 1], p)
        gcol_low = @view G_Low[:, i]
        gcol_up  = @view G_Up[:, i]

        for jt in Iterators.product(ntuple(k -> 0:e[k], p)...)
            coeff = 1.0
            for k in 1:p
                coeff *= cvecs[k][jt[k] + 1]
            end
            key = NTuple{p,Int8}(jt)

            col = get(key_to_col, key, 0)
            if col == 0
                ncols += 1
                col = ncols
                key_to_col[key] = col
                acc_low[key] = coeff .* collect(gcol_low)
                acc_up[key]  = coeff .* collect(gcol_up)
            else
                acc_low[key] .+= coeff .* gcol_low
                acc_up[key]  .+= coeff .* gcol_up
            end

            push!(contributions[i], (col, coeff))
        end
    end

    hnew = ncols
    Enew     = Matrix{Int}(undef, p, hnew)
    Gnew_Low = Matrix{T}(undef, n, hnew)
    Gnew_Up  = Matrix{T}(undef, n, hnew)
    for (key, col) in key_to_col
        Enew[:, col]     .= key
        Gnew_Low[:, col] .= acc_low[key]
        Gnew_Up[:, col]  .= acc_up[key]
    end

    project_GL = ProjectTo(G_Low)
    project_GU = ProjectTo(G_Up)

    function map_sp_pullback(Δ)
        Δ = unthunk(Δ)
        # Δ is a tangent for the returned tuple (Enew, Gnew_Low, Gnew_Up)
        dEnew, dGL_new, dGU_new = if Δ isa Tuple || Δ isa AbstractVector
            (Δ[1], Δ[2], Δ[3])
        else
            # Fallback: treat as zero if unexpected
            (NoTangent(), NoTangent(), NoTangent())
        end

        dGL_new = dGL_new isa AbstractZero ? nothing : unthunk(dGL_new)
        dGU_new = dGU_new isa AbstractZero ? nothing : unthunk(dGU_new)

        dG_Low = zeros(T, n, h)
        dG_Up  = zeros(T, n, h)

        @inbounds for i in 1:h
            for (col, coeff) in contributions[i]
                if dGL_new !== nothing
                    @views dG_Low[:, i] .+= coeff .* dGL_new[:, col]
                end
                if dGU_new !== nothing
                    @views dG_Up[:, i]  .+= coeff .* dGU_new[:, col]
                end
            end
        end

        return (NoTangent(),                 # function itself
                NoTangent(),                 # E (integer, non-diff)
                project_GL(dG_Low),          # G_Low
                project_GU(dG_Up))           # G_Up
    end

    return (Enew, Gnew_Low, Gnew_Up), map_sp_pullback
end



function to_sparse_polynomial(inter::CombinedPolyBernsteinInterval)
    @assert all(inter.orders .<= 2)
    num_polys = size(inter.Low,1)
    if all(inter.orders .== 1)
        # [0,1]: x^1 
        # [1,1]: x^0
        exponents = 1 .- inter.bern_terms[:, 1] 
        exponents = reshape(exponents, inter.n,inter.t)
    else # assumed to be 2
        # [0,0,1]: x^2 
        # [0,1/2,1]: x^1 
        # [1,1,1]: x^0
        exponents = 2 .-  2 .* inter.bern_terms[:, 2] 
        exponents = reshape(exponents, inter.n,inter.t)
    end


    E, G_Low, G_Up =  map_sp_to_correct_interval(exponents,inter.Low, inter.Up)
    return PolyInterval(
        SparsePolynomial(G_Low,E,collect(1:num_polys) ),
        SparsePolynomial(G_Up,E,collect(1:num_polys) ))


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
Construct a bernstein Polynomial repr of l + (u-l)x_i over the input domain [0,1]
"""
function init_one_mapped_coeff_mat(poly_idx,num_poly_vars, )
    coeff_mat = fill(1.0, num_poly_vars, 2)
    coeff_mat[poly_idx, :] = [0, 1.]
    return coeff_mat
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
    bern_terms_matrix  = ones(eltype(h.radius), (n+1)*n,2)
    coeff_matrix  = zeros(eltype(h.radius), num_polys ,n+1  )
    #@show coeff_matrix
    
    # one term x_i for each unfixed variable
    for x_i in 1:n 
	    bern_terms_matrix_idx = (x_i-1)*n 
	    bern_terms_matrix[bern_terms_matrix_idx + x_i , 1]= 0.
    end
    # one constant term (1) at the end. (1,1,1) does not need to be modified

    # set coefficient vectors 
    for poly_idx in eachindex(h.radius)
	    if unfixed_mask[poly_idx] # set to l + (u - l)*x 
	        x_index =  count(@view unfixed_mask[1:poly_idx]) 
	        coeff_matrix[poly_idx, x_index]  = 2*h.radius[poly_idx] 
	        coeff_matrix[poly_idx, n+1]  = low(h)[poly_idx]
	    else 
	        coeff_matrix[poly_idx, n+1]  = h.center[poly_idx]
	    end
    end

    return CombinedPolyBernsteinInterval(
	    copy(coeff_matrix),coeff_matrix,bern_terms_matrix,n+1,n,fill(1,n),Hyperrectangle(fill(0.5,n),fill(0.5,n))
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
    binom_old = binomial_tensor_cached(interval.orders)
    binom_new = binomial_tensor_cached(new_orders)
    C = repeat(binom_old,interval.t)
    C_rescale = repeat(binom_new,new_t)
    scaled = interval.bern_terms .* C 
    
    scaled_a = reduce(vcat, [
	repeat(scaled[(term_idx-1)*interval.n + 1 : term_idx*interval.n, :], interval.t)
	for term_idx in 1:interval.t
    ])

    scaled_b = repeat(scaled,interval.t)

    #conv = row_convolution_kernel(scaled_a,scaled_b)
    conv = batched_conv_rowwise(scaled_a,scaled_b)

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

    @assert all( <=(2) ,new_order )
    elevated_a =  (new_order != inter_a.orders ) ? elevate_all_to(inter_a,new_order) : inter_a
    elevated_b =  (new_order != inter_b.orders ) ? elevate_all_to(inter_b,new_order) : inter_b
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
max/min for one in all variables monoton increasing term in implicit repr. 
"""
function term_extrema(term, orders)
    left_prod = one(eltype(term))
    right_prod = one(eltype(term))
    for row_idx in axes(term,1)
        left_prod *= term[row_idx,1]
        right_prod *= term[row_idx,orders[row_idx]+1]
    end
    return min(left_prod, right_prod), max(left_prod, right_prod)
end
""" 
min diff along the j-th dimension of B_term. Requires term to be monotonically increasing in all variabels.
"""
function term_min_step_along_j(term,orders, j)
    d = diff(term[j,1:orders[j]+1])                 
    # if less than zero must be numerical error, therefore rounding is fine!
    min_step_j = max(minimum(d),0.0)

    other_factor = one(eltype(term)) 
    for m  in axes(term,1)
        m == j && continue
        other_factor *= term[m,1]
    end
    return min_step_j * abs(other_factor)
end

"""
width of a term: max - min 
"""
term_width(term,orders) = ((lo,hi) = term_extrema(term, orders); hi - lo)
function bound_algo(as,orders::Vector{Int},t::Int,j::Int,constant_terms,term_widths,non_constant_mask, inc_mask, dec_mask, width_dec_full, width_inc_full , all_diff_dec, all_diff_inc )
    l_j = orders[j]+1

    non_constant_mask = non_constant_mask .||  .!(constant_terms[:,j])
    constant_mask = .!non_constant_mask

    # exactly one term is non-constant with respect to x_j
    if count(non_constant_mask) == 1
        min_idx,max_idx =  as[non_constant_mask][1] > 0.0 ? (1, l_j ) : (l_j, 1)
        return min_idx:min_idx, max_idx:max_idx
    end
    # more than one non constant term therfore continue with monotonicity_test
    #increasing_mask = as[non_constant_mask] .>= 0

    inc_mask = inc_mask .&& non_constant_mask
    dec_mask = dec_mask .&& non_constant_mask

    if all(!,inc_mask)
        return l_j:l_j, 1:1
    end
    if all(!,dec_mask)
        return 1:1, l_j:l_j
    end
    
    width_dec = width_dec_full -  sum( term_widths[constant_mask] .* abs.(as[constant_mask]) )
    #diff_inc = sum(min_steps_j[increasing_mask,j] .* abs.(as[increasing_mask])  )
    diff_inc = all_diff_inc[j]
    
    @assert diff_inc >= 0  "$diff_inc , $(as[increasing_indices]),"

    if diff_inc > width_dec
        return   1:1,l_j:l_j
    end
    

    width_inc = width_inc_full -  sum( term_widths[constant_mask] .* abs.(as[constant_mask]) )
    #diff_dec = sum(min_steps_j[decreasing_mask,j] .* abs.(as[decreasing_mask])  )
    diff_dec = all_diff_dec[j]
    @assert diff_dec >= 0  "$diff_dec , $(as[decreasing_mask]) "
    if diff_dec > width_inc
        return   l_j:l_j,1:1
    end

    return 1:l_j, 1:l_j
end
global calls_0d=0
global calls_1d=0
global calls_2d=0
global calls_higher=0
function _eval_broadcast(bern_mat, as::Vector{TA}, t, n, S, nonscalar, dims::NTuple{K,Int}) where {K,TA}
    T = promote_type(eltype(bern_mat), TA)
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
        return (s, s)          
    end
    out = zeros(T, dims)
    for t_idx in 1:t
        a = as[t_idx]; iszero(a) && continue
        rows = get_term(t_idx, n)
        coeff = a
        @inbounds for m in 1:n
            length(S[m]) == 1 && (coeff *= bern_mat[rows[m], first(S[m])])
        end
        iszero(coeff) && continue
        factors = ntuple(K) do k          
            mk = nonscalar[k]
            sub = @view bern_mat[rows[mk], S[mk]]
            shape = ntuple(d -> d == k ? dims[k] : 1, K)
            reshape(sub, shape...)
        end
        out .+= coeff .* .*(factors...)
    end
    return extrema(out)
end

function ChainRulesCore.rrule(::typeof(_eval_broadcast),bern_mat,as::Vector{TA},t,n,S,nonscalar,dims::NTuple{K,Int}) where {K,TA}
        T = promote_type(eltype(bern_mat), TA)

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
        idx_min = idx_max = nothing   # kein Index nötig, alle Terme tragen bei
    else
        out = zeros(T, dims)
        for t_idx in 1:t
            a = as[t_idx]; iszero(a) && continue
            rows = get_term(t_idx, n)
            coeff = a
            @inbounds for m in 1:n
                length(S[m]) == 1 && (coeff *= bern_mat[rows[m], first(S[m])])
            end
            iszero(coeff) && continue
            factors = ntuple(K) do k
                mk = nonscalar[k]
                sub = @view bern_mat[rows[mk], S[mk]]
                shape = ntuple(d -> d == k ? dims[k] : 1, K)
                reshape(sub, shape...)
            end
            out .+= coeff .* .*(factors...)
        end
        min_val, idx_min = findmin(out)
        max_val, idx_max = findmax(out)
    end

    result = K == 0 ? (s, s) : (min_val, max_val)

    function pullback(Δ)
        Δmin, Δmax = Δ
        Δas = zero(as)

        function accumulate!(Δas, idx, Δscalar)
            iszero(Δscalar) && return
            for t_idx in 1:t
                a = as[t_idx]
                rows = get_term(t_idx, n)
                scalar_factor = one(T)
                @inbounds for m in 1:n
                    length(S[m]) == 1 && (scalar_factor *= bern_mat[rows[m], first(S[m])])
                end
                nonscalar_factor = one(T)
                if K > 0
                    @inbounds for k in 1:K
                        mk = nonscalar[k]
                        nonscalar_factor *= bern_mat[rows[mk], S[mk][idx[k]]]
                    end
                end
                Δas[t_idx] += Δscalar * scalar_factor * nonscalar_factor
            end
        end

        if K == 0
            for t_idx in 1:t
                rows = get_term(t_idx, n)
                p = one(T)
                @inbounds for m in 1:n
                    p *= bern_mat[rows[m], first(S[m])]
                end
                Δas[t_idx] += (Δmin + Δmax) * p   # min==max, add both Δ 
            end
        else
            accumulate!(Δas, Tuple(idx_min), Δmin)
            accumulate!(Δas, Tuple(idx_max), Δmax)
        end

        return (NoTangent(), NoTangent(), Δas, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return result, pullback
end

"""
computes all coefficients in S and returns mininmun , maximums
"""
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
"""
compute min/ max for each term and add them all
"""
function imp_fast_bounds(as::AbstractArray,bern_mat::AbstractArray,t::Int,orders)
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
    return b_min, b_max
end


function faster_exact_bounds(as,bern_mat::AbstractArray,orders::AbstractArray,t::Int,constant_terms, term_widths, min_steps_j; method=SmithBoundsOverapproximate, threshold=5)
    n = length(orders)
    S_max =   Vector{UnitRange{Int64}}(undef, n)
    S_min = Vector{UnitRange{Int64}}(undef, n)
    # expand bern_mat 

    non_constant_alphas = .!( isapprox.(as,0))
    inc_mask = non_constant_alphas .&& as .> 0 
    dec_mask = non_constant_alphas .&& .!inc_mask

    width_dec_full = sum( term_widths[dec_mask] .* abs.(as[dec_mask]) )
    width_inc_full = sum( term_widths[inc_mask] .* abs.(as[inc_mask]) )

    #diff_inc = sum(min_steps_j[increasing_mask,j] .* abs.(as[increasing_mask])  )

    all_diff_inc = min_steps_j' * (as .* inc_mask) # we can ignore constant terms here because min_steps_j is zero for them 

    all_diff_dec = min_steps_j' * (abs.(as) .* dec_mask) # we can ignore constant terms here because min_steps_j is zero for them 

    @ignore_derivatives for x_i in 1:n
	    S_min[x_i],S_max[x_i] = bound_algo(as,orders,t,x_i,constant_terms,term_widths, non_constant_alphas, inc_mask, dec_mask ,width_dec_full ,width_inc_full, all_diff_dec, all_diff_inc)
    end
    min_possibilites = prod(length, S_min)
    if min_possibilites > threshold || min_possibilites < 0 # check for overflow
        if method == SmithBoundsOverapproximate
            b_min, b_max = imp_fast_bounds(as,bern_mat,t,orders)
        else 
            b_min = -Inf
            b_max = Inf
        end
    elseif S_min == S_max 
        b_min, b_max = evaluate_reduced_tensor(bern_mat,as,t,n,S_min)
    else
        b_min,_ = evaluate_reduced_tensor(bern_mat,as,t,n,S_min)
        _, b_max = evaluate_reduced_tensor(bern_mat,as,t,n,S_max)
    end

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
    term_widths = [term_width((@view bern_mat[get_term(t_idx,n),:]) ,orders) for t_idx in 1:t ]
    min_steps_j = [term_min_step_along_j((@view bern_mat[get_term(t_idx, n), :]), orders, j)
               for t_idx in 1:t, j in 1:n]
    return constant_terms,term_widths, min_steps_j
end

function bounds(interval::CombinedPolyBernsteinInterval;  method=Overapproximate, threshold=-1)
    num_p = size(interval.Low,1)
    T = eltype(interval.Low)
    llbs = Vector{T}()
    lubs = Vector{T}()
    ulbs = Vector{T}()
    uubs = Vector{T}()
    sizehint!(llbs,num_p)
    sizehint!(lubs,num_p)
    sizehint!(ulbs,num_p)
    sizehint!(uubs,num_p)

    
    if method == Overapproximate 
	    for p_idx in axes(interval.Low,1)
	        low_poly = get_poly_low(p_idx,interval)
	        up_poly = get_poly_up(p_idx,interval)
	        llbsi , lubsi  = quadrant_ibf_minmax(low_poly,interval.t,interval.orders)#,monomon_coefficients, inv(stack(monomon_coefficients)))
	        ulbsi, uubsi = quadrant_ibf_minmax(up_poly,interval.t,interval.orders)#,monomon_coefficients, inv(stack(monomon_coefficients)))

		llbs = [llbs..., llbsi]
		lubs = [lubs..., lubsi]
		ulbs = [ulbs..., ulbsi]
		uubs = [uubs..., uubsi]
	    end
	    return llbs,lubs, ulbs, uubs
    end

    if method == SmithBoundsMonomon
        poly_interval = to_sparse_polynomial(interval) 
        llbs_sp , lubs_sp = bounds(poly_interval.Low)
        ulbs_sp , uubs_sp = bounds(poly_interval.Up)
    end
    constant_terms,term_widths , min_steps_j = @ignore_derivatives precompute(interval.bern_terms, interval.t,interval.orders)

    unique_as  =  Vector{Tuple{Int64,Bool}}()
    low_evaluated_poly = Array{Int64}(undef,size(interval.Low,1))
    @ignore_derivatives for i  in axes(interval.Low,1)
        idx = findfirst( x -> isapprox((@view interval.Low[i,:]), (@view interval.Low[x[1],:]) ) , unique_as)
        if isnothing(idx)
            push!(unique_as,(i,true))
            low_evaluated_poly[i] =size(unique_as,1)
        else 
            low_evaluated_poly[i] = idx
        end
    end

    up_evaluated_poly =  Array{Int64}(undef,size(interval.Up,1))
    @ignore_derivatives for i  in axes(interval.Up,1)
        idx = findfirst( x -> isapprox((@view interval.Up[i,:]),( x[2] ? (@view interval.Low[x[1],:]) : @view interval.Up[x[1],:])) , unique_as)
        if isnothing(idx)
            push!(unique_as,(i,false))
            up_evaluated_poly[i] =size(unique_as,1)
        else 
            up_evaluated_poly[i] = idx
        end
    end

    println("saved: $(2*size(interval.Low,1) - size(unique_as,1) )")

    unique_bounds = Zygote.Buffer(Array{Tuple{Float64,Float64}}(undef,1),size(unique_as,1))
    for i in eachindex(unique_as)
        println("$i")
        p_idx, is_lower = unique_as[i]
        if is_lower
            unique_bounds[i] = faster_exact_bounds(interval.Low[p_idx,:] ,interval.bern_terms,interval.orders,interval.t,constant_terms,term_widths,min_steps_j; method,  threshold) 
        else
            unique_bounds[i] = faster_exact_bounds(interval.Up[p_idx,:] ,interval.bern_terms,interval.orders,interval.t,constant_terms,term_widths,min_steps_j; method,threshold) 
        end
    end



    for p_idx in axes(interval.Low,1)
    
        llbsi , lubsi  = unique_bounds[low_evaluated_poly[p_idx]]
        ulbsi , uubsi = unique_bounds[up_evaluated_poly[p_idx]]

        if method == SmithBoundsMonomon
            llbs = [llbs..., max(llbsi, llbs_sp[p_idx])]
            lubs = [lubs..., min(lubsi, lubs_sp[p_idx])]
            ulbs = [ulbs..., max(ulbsi, ulbs_sp[p_idx])]
            uubs = [uubs..., min(uubsi, uubs_sp[p_idx])]
        else 
            llbs = [llbs..., llbsi]
            lubs = [lubs..., lubsi]
            ulbs = [ulbs..., ulbsi]
            uubs = [uubs..., uubsi]

        end
    end
    return llbs,lubs,ulbs,uubs

end

"""
Calculates concrete bounds for A*s + b for BernsteinPoly s with common generators.
"""
function bounds(A::AbstractMatrix, b::AbstractVector, s::CombinedPolyBernsteinInterval; method=Overapproximate, threshold=-1)
    mapped_interval = interval_map(
        min.(0, A),
        max.(0, A),
        s,
        b,
    )
    ll, lu ,ul, uu = bounds(mapped_interval; method , threshold)
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
	t_idx = findfirst(x -> isapprox(x , term; atol=1e-14),unique_terms)
	if isnothing(t_idx)
	    unique_terms = [unique_terms...,copy(term)]
	    Low = [Low zeros(eltype(Low),size(Low,1))]
	    Up = [Up zeros(eltype(Up),size(Up,1))]
	    t_idx = length(unique_terms)
	end
	println("mapping $old_term_idx to  $t_idx ")
	#Low[:,t_idx] .+=  inter.Low[:, old_term_idx]
	Low = hcat(Low[:,1:t_idx-1], 
		    Low[:,t_idx] .+ inter.Low[:,old_term_idx],
		    Low[:,t_idx+1:end])
	Up = hcat(Up[:,1:t_idx-1], 
		    Up[:,t_idx] .+ inter.Up[:,old_term_idx],
		    Up[:,t_idx+1:end])
	#Up[:,t_idx] .+=  inter.Up[:, old_term_idx]
    end
    return CombinedPolyBernsteinInterval(Low,Up,vcat(unique_terms...),length(unique_terms),inter.n,inter.orders,inter.X,inter.lbs,inter.ubs)
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
