using LazySets
using DynamicPolynomials


struct CombinedPolyBernsteinInterval{N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O},VN<:AbstractArray{N}}
    Low::TN # each row of the matrix corresponds to the term coefficients for one Low polynomial
    Up::TN # each row of the matrix corresponds to the term coefficients for one Upper polynomial
    bern_terms_coeffs::BitArray # bit array containing 0 if the entry in bern_terms is zero, 1 otherwise
    bern_terms_2_exp::Matrix{Int8} # matrix containing the log_2(coeff) or undefined for 0 
    t::M # number of terms
    n::M # input dimension
    order::Int8 # orders of the variables
    X::Hyperrectangle{N}
    lbs::Vector{VN}  # (vector of vectors) lower bounds for intermediate values
    ubs::Vector{VN}  # (vector of vectors) upper bounds for intermediate values
end

function map_to_2_pow(x::T)::Int8 where {T<:Number}
    if isapprox(0.5, x)
        return -1
    elseif isapprox(1, x)
        return 0
    else
        return -127
    end
end

function normal_two_compressed(bern_terms::TN) where {N<:Number,TN<:AbstractArray{N}}
    # 0.5 -> 1 * 2^-1
    # 1 -> 1 * 2^0
    # 0 -> 0 * 2^{undef}
    coeff = .!isapprox.(bern_terms, 0.0; atol=1e-13)
    two_exponents = map_to_2_pow.(bern_terms)
    return coeff, two_exponents
end

function compressed_to_normal(inter::CombinedPolyBernsteinInterval)
    return  inter.bern_terms_coeffs .* ldexp.(1.0, inter.bern_terms_2_exp)
end

function CombinedPolyBernsteinInterval(Low::TN, Up::TN, bern_terms::TN, t::M, n::M, order::Int8, X::Hyperrectangle{N}) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    zeros, two_exponents = normal_two_compressed(bern_terms)
    return CombinedPolyBernsteinInterval(Low, Up, zeros, two_exponents, t, n, order, X, Vector{Vector{N}}(), Vector{Vector{N}}())
end
function CombinedPolyBernsteinInterval(Low::TN, Up::TN, bern_terms_coeffs::BitArray, bern_terms_2_exp::Matrix{Int8}, t::M, n::M, order::Int8, X::Hyperrectangle{N}) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    return CombinedPolyBernsteinInterval(Low, Up, bern_terms_coeffs, bern_terms_2_exp, t, n, order, X, Vector{Vector{N}}(), Vector{Vector{N}}())
end

# fix coefficient table for x^e = ((x'+1)/2)^e, e ∈ {0,1,2} padded with zeros, for same Type
const COEFFS2::NTuple{3,NTuple{3,Float64}} = (
    (1.0, 0.0, 0.0),            # e = 0: (x+1)/2^0 = 1*x^0
    (0.5, 0.5, 0.0),        # e = 1: (x+1)/2 = 1/2 x^0 + 1/2x^1 
    (0.25, 0.5, 0.25), # e = 2: (x/2 + 1/2)^2 = x^2/4 + 2* x/2 * 1/2 + 1/4 = 1/4*x^0 + 1/2*x^1 + 1/4*x^2
)
function map_sp_to_correct_interval(E::AbstractArray{Int8}, G_Low::TN, G_Up::TN) where {T<:Number,TN<:AbstractArray{T}}
    p, h = size(E)
    n = size(G_Low, 1)
    @assert size(G_Low, 2) == h "E and G_Low need the same amount of columns"
    @assert size(G_Up, 2) == h "E and G_Up need the same amount of columns"
    @assert all(x -> 0 <= x <= 2, E) "this function requires degree <= 2 per variable."

    acc_low = Dict{Vector{Int8},Vector{T}}()
    acc_up = Dict{Vector{Int8},Vector{T}}()
    sizehint!(acc_low, min(h * 4, 3^min(p, 20)))
    sizehint!(acc_up, min(h * 4, 3^min(p, 20)))
    e = Vector{Int8}(undef, p)
    cvecs = Vector{NTuple{3,Float64}}(undef, p)

    @inbounds for i in 1:h
        for k in 1:p
            e[k] = E[k, i]
            cvecs[k] = COEFFS2[e[k]+1]
        end

        gcol_low = @view G_Low[:, i]
        gcol_up = @view G_Up[:, i]

        idx = zeros(Int, p)
        done = false
        #for jt in Iterators.product(ntuple(k -> 0:e[k], p)...)
        while !done
            coeff = 1.0
            for k in 1:p
                coeff *= cvecs[k][idx[k]+1]
            end
            #key = NTuple{p,Int8}(jt)


            v_low = get(acc_low, idx, nothing)
            if v_low === nothing
                acc_low[copy(idx)] = coeff .* collect(gcol_low)
                acc_up[copy(idx)] = coeff .* collect(gcol_up)
            else
                v_up = get(acc_up, idx, nothing)
                v_low .+= coeff .* gcol_low
                v_up .+= coeff .* gcol_up
            end
	    # grow idx "digit by digit" 
            k = 1
            while k <= p
                if idx[k] < e[k]
                    idx[k] += 1
                    break
                else
                    idx[k] = 0
                    k += 1
                end
            end
            k > p && break
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
    for (col, (j, g)) in enumerate(acc_up)
        Gnew_Up[:, col] .= g
    end

    return Enew, Gnew_Low, Gnew_Up
end
function ChainRulesCore.rrule(::typeof(map_sp_to_correct_interval), E::AbstractArray{Int8}, G_Low::TN, G_Up::TN) where {T<:Number,TN<:AbstractArray{T}}
    p, h = size(E)
    n = size(G_Low, 1)

    acc_low = Dict{Vector{Int8},Vector{T}}()
    acc_up  = Dict{Vector{Int8},Vector{T}}()
    key_to_col = Dict{Vector{Int8},Int}()
    contributions = [Tuple{Int,Float64}[] for _ in 1:h]

    ncols = 0
    e = Vector{Int8}(undef, p)
    cvecs = Vector{NTuple{3,Float64}}(undef, p)

    @inbounds for i in 1:h
        for k in 1:p
            e[k] = E[k, i]
            cvecs[k] = COEFFS2[e[k]+1]
        end
        gcol_low = @view G_Low[:, i]
        gcol_up  = @view G_Up[:, i]

        idx = zeros(Int8, p)
        while true
            coeff = 1.0
            for k in 1:p
                coeff *= cvecs[k][idx[k]+1]
            end

            col = get(key_to_col, idx, 0)
            if col == 0
                ncols += 1
                col = ncols
                key = copy(idx)
                key_to_col[key] = col
                acc_low[key] = coeff .* collect(gcol_low)
                acc_up[key]  = coeff .* collect(gcol_up)
            else
                acc_low[idx] .+= coeff .* gcol_low
                acc_up[idx]  .+= coeff .* gcol_up
            end
            push!(contributions[i], (col, coeff))

            k = 1
            while k <= p
                if idx[k] < e[k]
                    idx[k] += 1
                    break
                else
                    idx[k] = 0
                    k += 1
                end
            end
            k > p && break
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
        dEnew, dGL_new, dGU_new = if Δ isa Tuple || Δ isa AbstractVector
            (Δ[1], Δ[2], Δ[3])
        else
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

        return (NoTangent(), NoTangent(), project_GL(dG_Low), project_GU(dG_Up))
    end

    return (Enew, Gnew_Low, Gnew_Up), map_sp_pullback
end



function to_sparse_polynomial(inter::CombinedPolyBernsteinInterval{N,M,O}) where {N<:Number,M<:Integer,O<:Number}
    @assert inter.order <= 2
    num_polys = size(inter.Low, 1)
    exponents = @ignore_derivatives begin
	exp = Matrix{Int8}(undef, 1, inter.n*inter.t)
	if inter.order == 1
    	    # [0,1]: x^1
    	    # [1,1]: x^0
    	    exp[1, :] .=  Int8.(1 .- inter.bern_terms_coeffs[:, 1])  #Int.(1 .- inter.bern_terms[:, 1] )
    	else # assumed to be 2
    	    # [0,0,1]: x^2
    	    # [0,1/2,1]: x^1
    	    # [1,1,1]: x^0
    	    exp[1, :] .= Int8.(2 .- (inter.bern_terms_coeffs[:, 2] .* ldexp.(1.0, inter.bern_terms_2_exp[:, 2] .+ 1)))
    	end
    	reshape(exp, inter.n, inter.t)
    end

    E, G_Low, G_Up = map_sp_to_correct_interval(exponents, inter.Low, inter.Up)
    return PolyInterval(
        SparsePolynomial(G_Low, E, collect(1:num_polys)),
        SparsePolynomial(G_Up, E, collect(1:num_polys)))


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
    C = zeros(Float64, length(polynoms), 0)   # num_polys×0 matrix
    # get all the terms and the Coefficient matrix
    all_terms = []
    for (poly_idx, polynom) in enumerate(polynoms)
        for t in terms(polynom)
            t_idx = findfirst(x -> x == t, all_terms)
            if isnothing(t_idx)
                push!(all_terms, t)
                C = [C zeros(eltype(C), size(C, 1))]
                t_idx = length(all_terms)
            end
            C[poly_idx, t_idx] += DynamicPolynomials.coefficient(t)
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


function with_coeffs(row, bern_terms, n)
    scales = [
        iszero((i - 1) % n) ? row[(i-1)÷n+1] : one(eltype(bern_terms))
        for i in axes(bern_terms, 1)
    ]
    return scales .* bern_terms
end
function get_poly_low(poly_idx::Int, interval::CombinedPolyBernsteinInterval)
    return with_coeffs(interval.Low[poly_idx, :], interval.bern_terms, interval.n)
end
function get_poly_up(poly_idx::Int, interval::CombinedPolyBernsteinInterval)
    return with_coeffs(interval.Up[poly_idx, :], interval.bern_terms, interval.n)
end
"""
Construct a bernstein Polynomial repr of l + (u-l)x_i over the input domain [0,1]
"""
function init_one_mapped_coeff_mat(poly_idx, num_poly_vars,)
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
    bern_terms_twoExps = zeros(Int8, (n+1)*n, 2)
    bern_terms_coeffs = trues((n+1)*n, 2)
    coeff_matrix = zeros(eltype(h.radius), num_polys, n+1)

    # one term x_i for each unfixed variable
    for x_i in 1:n
        bern_terms_matrix_idx = (x_i-1)*n
        bern_terms_coeffs[bern_terms_matrix_idx+x_i, 1] = false
    end
    # one constant term (1) at the end. (1,1,1) does not need to be modified

    # set coefficient vectors 
    for poly_idx in eachindex(h.radius)
        if unfixed_mask[poly_idx] # set to l + (u - l)*x 
            x_index = count(@view unfixed_mask[1:poly_idx])
            coeff_matrix[poly_idx, x_index] = 2*h.radius[poly_idx]
            coeff_matrix[poly_idx, n+1] = low(h)[poly_idx]
        else
            coeff_matrix[poly_idx, n+1] = h.center[poly_idx]
        end
    end

    return CombinedPolyBernsteinInterval(
        copy(coeff_matrix), coeff_matrix, bern_terms_coeffs, bern_terms_twoExps, n+1, n, Int8(1), Hyperrectangle(fill(0.5, n), fill(0.5, n))
    )
end

function constant_term(type, n, orders)

    term = zeros(type, n, maximum(orders)+1)
    for i in axes(term, 1)
        term[i, 1:(orders[i]+1)] .= one(type)
    end
    return term

end

function constant_term(n, order)

    term_coeffs = fill(true, n, order+1)
    term_2_exp = fill(0, n, order+1)
    return term_coeffs, term_2_exp

end
function translate(interval::CombinedPolyBernsteinInterval, b)
    return translate(interval, b, b)
end

function translate(interval::CombinedPolyBernsteinInterval, b_low, b_up)
    #  look for constant term in the polynomial and increase coefficents at the index: 
    for i in 1:interval.t
        term_two_exp = @view interval.bern_terms_2_exp[((i-1)*interval.n+1):(i*interval.n), :]
        term_coeffs = @view interval.bern_terms_coeffs[((i-1)*interval.n+1):(i*interval.n), :]
        if all(term_two_exp .== 0 .&& term_coeffs)
            # constant term found
            low = hcat(
                interval.Low[:, 1:(i-1)],
                interval.Low[:, i]+b_low,
                interval.Low[:, (i+1):end],
            )
            up = hcat(
                interval.Up[:, 1:(i-1)],
                interval.Up[:, i]+b_up,
                interval.Up[:, (i+1):end],
            )
            return CombinedPolyBernsteinInterval(low, up, copy(interval.bern_terms_coeffs), copy(interval.bern_terms_2_exp), interval.t, interval.n, interval.order, interval.X)
        end
    end
    # no constant term found therefore add it at the end of the matrix
    term_coeffs, term_2_exp = constant_term(interval.n, interval.order)
    new_bterm_coeffs = vcat(interval.bern_terms_coeffs, term_coeffs)
    new_terms_2_exp = vcat(interval.bern_terms_2_exp, term_2_exp)
    return CombinedPolyBernsteinInterval([interval.Low b_low], [interval.Up b_up], new_bterm_coeffs, new_terms_2_exp, interval.t+1, interval.n, interval.order, interval.X)

end

"""
Interval map overapproximating an affine map Wx + b for x ∈ I 

W⁻ - (matrix) negative weights
W⁺ - (matrix) positive weights
I  - (BernsteinInterval) bernsteinInterval
b  - (vector) bias
"""
function interval_map(W⁻, W⁺, I::CombinedPolyBernsteinInterval, b; use_memory_optimizations=true)
    new_low = W⁻ * I.Up + W⁺ * I.Low
    new_up = W⁻ * I.Low + W⁺ * I.Up
    return translate(CombinedPolyBernsteinInterval(new_low, new_up, I.bern_terms_coeffs, I.bern_terms_2_exp, I.t, I.n, I.order, I.X), b)
end


function square(Low::TC, Up::TC, bern_terms::TN; use_memory_optimizations=true) where {N<:Number,M<:Number,TN<:AbstractArray{N},TC<:AbstractArray{M}}
    t = size(Low, 2)
    new_t = t^2
    cur_order = size(bern_terms, 2) - 1
    n = size(bern_terms, 1) ÷ t
    new_order = 2 .* cur_order
    binom_old = binomial_tensor_cached(cur_order)
    binom_new = binomial_tensor_cached(new_order)
    C = binom_old
    C_rescale = binom_new
    scaled = bern_terms .* C

    scaled_a = reduce(vcat, [
        repeat(scaled[((term_idx-1)*n+1):(term_idx*n), :], t)
        for term_idx in 1:t
    ])

    scaled_b = repeat(scaled, t)

    #conv = row_convolution_kernel(scaled_a,scaled_b)
    conv = batched_conv_rowwise(scaled_a, scaled_b)

    new_terms = conv ./ C_rescale

    n_rows, n_cols = size(Low)

    # (n_rows, n_cols, 1) .* (n_rows, 1, n_cols) -> (n_rows, n_cols, n_cols), dann reshape
    low_outer = reshape(Low, n_rows, n_cols, 1) .* reshape(Low, n_rows, 1, n_cols)
    new_low = reshape(low_outer, n_rows, n_cols^2)

    up_outer = reshape(Up, n_rows, n_cols, 1) .* reshape(Up, n_rows, 1, n_cols)
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
    #new_zeros , new_powers_of_two = normal_two_compressed(new_terms) 
    return new_low, new_up, new_terms
    #if use_memory_optimizations 
    ret#urn combine_terms(CombinedPolyBernsteinInterval(new_low,new_up,new_zeros,new_powers_of_two,new_t,interval.n,new_order,interval.X))
    #end
    #return CombinedPolyBernsteinInterval(new_low,new_up,new_zeros,new_powers_of_two,new_t,interval.n,new_order,interval.X)
end
function elevate_all_to(bern_terms, new_order::Int64)
    bern_term_rows = size(bern_terms, 1)
    order = size(bern_terms, 2) - 1
    diff = new_order - order
    bin_tensor_row = binomial_tensor_cached(order)


    res = bern_terms .* bin_tensor_row


    C_diff = repeat(binomial_tensor_cached(diff), bern_term_rows, 1)

    C_new_orders = repeat(binomial_tensor_cached(new_order), bern_term_rows, 1)

    result = row_convolution_kernel(res, C_diff) ./ C_new_orders

    return result
end
function add(low_a, up_a, bern_terms_a, low_b, up_b, bern_terms_b; use_memory_optimizations=true)
    order_a = size(bern_terms_a, 2) - 1
    order_b = size(bern_terms_b, 2) - 1
    new_order = max(order_a, order_b)

    elevated_a = (new_order != order_a) ? elevate_all_to(bern_terms_a, new_order) : bern_terms_a
    elevated_b = (new_order != order_b) ? elevate_all_to(bern_terms_b, new_order) : bern_terms_b

    return [low_a low_b], [up_a up_b], [elevated_a; elevated_b]

    #if use_memory_optimizations 
    # return combine_terms(CombinedPolyBernsteinInterval( 
    # [elevated_a.Low elevated_b.Low], [elevated_a.Up elevated_b.Up], 
    # [elevated_a.bern_terms_coeffs; elevated_b.bern_terms_coeffs] , [elevated_a.bern_terms_2_exp ; elevated_b.bern_terms_2_exp], elevated_a.t + elevated_b.t, elevated_a.n, new_order, elevated_a.X ))
    #end
    #return CombinedPolyBernsteinInterval( [elevated_a.Low elevated_b.Low], [elevated_a.Up elevated_b.Up],[elevated_a.bern_terms_coeffs; elevated_b.bern_terms_coeffs], [elevated_a.bern_terms_2_exp ; elevated_b.bern_terms_2_exp], elevated_a.t + elevated_b.t, elevated_a.n, new_order, elevated_a.X )
end


"""
Computes a one-dimensional quadratic map for each input dimension.
I.e. yᵢ =  qᵢxᵢ² for each input dimension i
"""
function quadratic_map_1d(qs_low, qs_up, Low, Up, bern_terms; use_memory_optimizations=true)
    quad_Up, quad_Low, quad_terms = square(Low, Up, bern_terms; use_memory_optimizations)
    return quad_Low .* qs_low, quad_Up .* qs_up, quad_terms
end


"""
Computes a one-dimensional quadratic function for each input dimension.
I.e. yᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ for each input dimension i
"""
function quadratic_propagation(a_low, b_low, c_low, a_up, b_up, c_up, inter::CombinedPolyBernsteinInterval; use_memory_optimizations=true)

    bern_terms = compressed_to_normal(inter)
    quad_low, quad_up, quad_terms = quadratic_map_1d(a_low, a_up, inter.Low, inter.Up, bern_terms; use_memory_optimizations)
    lin_low_coeffs = inter.Low .* b_low
    lin_up_coeffs = inter.Up .* b_up

    sum_low, sum_up, sum_terms = add(lin_low_coeffs, lin_up_coeffs, bern_terms, quad_low, quad_up, quad_terms)

    coeffs, twoExps = normal_two_compressed(sum_terms)

    res_inter = combine_terms(CombinedPolyBernsteinInterval(sum_low, sum_up, coeffs, twoExps, size(sum_low, 2), inter.n, Int8(size(coeffs, 2)-1), inter.X, inter.lbs, inter.ubs))

    return translate(res_inter, c_low, c_up)

end
function unique_idx(term, orders, j)
    #n = length(orders)
    #left_prod = one(eltype(term))
    #right_prod = one(eltype(term))
    #for row_idx in axes(term,1)
    #    left_prod *= term[row_idx,1]
    #    right_prod *= term[row_idx,orders[row_idx]]
    #end
    if all(term[1, :] .<= 0.0)
        return orders[j] + 1, 1
    end
    return 1, orders[j] + 1
end

"""
max/min for one in all variables monoton increasing term in implicit repr. 
"""
function term_extrema(term, orders)
    left_prod = one(eltype(term))
    right_prod = one(eltype(term))
    for row_idx in axes(term, 1)
        left_prod *= term[row_idx, 1]
        right_prod *= term[row_idx, orders[row_idx]+1]
    end
    return min(left_prod, right_prod), max(left_prod, right_prod)
end

""" 
min diff along the j-th dimension of B_term. Requires term to be monotonically increasing in all variabels.
"""
function term_min_step_along_j(term, orders, j)

    d = diff(term[j, 1:(orders[j]+1)])
    # if less than zero must be numerical error, therefore rounding is fine!
    min_step_j = max(minimum(d), 0.0)

    other_factor = one(eltype(term))
    for m in axes(term, 1)
        m == j && continue
        other_factor *= term[m, 1]
    end
    return min_step_j * abs(other_factor)
end

function term_min_step_along_j(coeffs, twoExps, n, order, j)::Float64

    if order == 1
        min_step_j = 1.0 - coeffs[j, 1]
    elseif order == 2
        min_step_j = coeffs[j,2] && twoExps[j, 2] == -1 ? 0.5 : 0
    end
    if min_step_j == 0
        return 0.
    end

    other_factor = reduce(&, coeffs[axes(coeffs, 1) .!= j, 1])
    return min_step_j * other_factor
end

"""
width of a term: max - min 
"""
term_width(term, orders) = ((lo, hi)=term_extrema(term, orders); hi - lo)
function bound_algo(as, order::Int8, t::Int, j::Int, constant_terms, term_widths, non_constant_mask, inc_mask, dec_mask, width_dec_full, width_inc_full, all_diff_dec, all_diff_inc)
    l_j = order+1

    non_constant_mask = non_constant_mask .&& .!(constant_terms[:, j])
    constant_mask = .!non_constant_mask

    # exactly one term is non-constant with respect to x_j
    if count(non_constant_mask) == 1
        min_idx, max_idx = as[non_constant_mask][1] > 0.0 ? (1, l_j) : (l_j, 1)
        return min_idx:min_idx, max_idx:max_idx
    end
    # more than one non constant term therfore continue with monotonicity_test
    #increasing_mask = as[non_constant_mask] .>= 0


    if all(!, inc_mask .&& non_constant_mask)
        return l_j:l_j, 1:1
    end
    if all(!, dec_mask .&& non_constant_mask)
        return 1:1, l_j:l_j
    end

    width_dec = width_dec_full - sum(term_widths[dec_mask .& constant_terms[:, j]] .* abs.(as[dec_mask .& constant_terms[:, j]]))
    #@assert width_dec == width_dec_full "$width_dec != $width_dec_full"
    #diff_inc = sum(min_steps_j[increasing_mask,j] .* abs.(as[increasing_mask])  )
    diff_inc = all_diff_inc[j]

    @assert diff_inc >= 0 "$diff_inc , $(all_diff_inc),"

    if diff_inc > width_dec
        return 1:1, l_j:l_j
    end


    width_inc = width_inc_full - sum(term_widths[constant_terms[:, j] .& inc_mask] .* abs.(as[constant_terms[:, j] .& inc_mask]))
    #diff_dec = sum(min_steps_j[decreasing_mask,j] .* abs.(as[decreasing_mask])  )
    diff_dec = all_diff_dec[j]
    @assert diff_dec >= 0 "$diff_dec , $(as[dec_mask]) "
    if diff_dec > width_inc
        return l_j:l_j, 1:1
    end

    return 1:l_j, 1:l_j
end
global calls_0d=0
global calls_1d=0
global calls_2d=0
global calls_higher=0

"""
Build the (t × prod(dims)) matrix M such that, for a given `as` vector sharing this S,
    out = reshape(M' * as, dims)
i.e. M[t_idx, :] holds the flattened outer product of coeff .* 2^exp for term t_idx,

"""
function build_M(coeffs::BitMatrix, twoExps::Matrix{Int8}, t::Int, n::Int, S, nonscalar, scalar_dims, dims::NTuple{K,Int}) where {K}
    total = prod(dims)
    #Mfloat = Matrix{Float64}(undef,t,total)
    Mfloat = zeros(t,total)


    # scratch buffers reused across t_idx to avoid per-term allocation
    for t_idx in 1:t
        rows = get_term(t_idx, n)

        scalar_zero = false
	# scalar dimensions are always 1 or l , therfore they are 0 or 2^0 
        @inbounds for m in scalar_dims
            r, c = rows[m], first(S[m])
            if !coeffs[r, c]
                scalar_zero = true
                break
            end
        end
        if scalar_zero
            # whole row of M stays zero
            continue
        end

        skip_term = false
        @inbounds for k in 1:K
            mk = nonscalar[k]
            if !any(@view coeffs[rows[mk], S[mk]])
                skip_term = true
                break
            end
        end
        skip_term && continue

        exp_factors = ntuple(K) do k
            mk = nonscalar[k]
            sub = @view twoExps[rows[mk], S[mk]]
            shape = ntuple(d -> d == k ? dims[k] : 1, K)
            reshape(sub, shape...)
        end
        coeffs_factors = ntuple(K) do k
            mk = nonscalar[k]
            sub = @view coeffs[rows[mk], S[mk]]
            shape = ntuple(d -> d == k ? dims[k] : 1, K)
            reshape(sub, shape...)
        end

	
        Mfloat_row = reshape(@view(Mfloat[t_idx, :]), dims)
	Mfloat_row .= (.&(coeffs_factors...)) .*  ldexp.(1.0, .+(exp_factors...))

    end

    return Mfloat 
end
const M_CACHE = Dict{Any, Matrix{Float64}}()   # keyed on (objectid(coeffs), S) or similar — see note below

function evaluate_reduced_tensor_cached(coeffs::BitMatrix, twoExps::Matrix{Int8}, as::Vector{TA}, t, n, S) where {TA}
    lens = [length(S[m]) for m in 1:n]
    nonscalar = [m for m in 1:n if lens[m] > 1]
    scalar_dims = [m for m in 1:n if lens[m] == 1]
    K = length(nonscalar)
    dims = ntuple(k -> lens[nonscalar[k]], Val(K))

    if K == 0
        # scalar case: no benefit from M, just do the direct sum (as before)
        return _eval_broadcast(coeffs, twoExps, as, t, n, S, nonscalar, dims)
    end

    key = (objectid(coeffs), objectid(twoExps), S)   # see caching-key note below
    M = get!(M_CACHE, key) do
        build_M(coeffs, twoExps, t, n, S, nonscalar, scalar_dims, dims)
    end

    out_flat = M' * as              # gemv: (prod(dims) × t) * (t) -> prod(dims)
    return extrema(out_flat)
end
function eval_scalar(coeffs::BitMatrix, twoExps::Matrix{Int8}, as::Vector{TA}, t, n,S ) where {TA}
    s = zero(TA)
    for t_idx in 1:t
        a = as[t_idx]
        iszero(a) && continue
        rows = get_term(t_idx, n)
        exp_acc = 0
        zero_hit = false
        @inbounds for m in 1:n
            r,c = rows[m], first(S[m])
            if !coeffs[r,c]
                zero_hit = true 
                break 
            end
            exp_acc += twoExps[r,c]
        end
        zero_hit && continue
        s += ldexp(a,exp_acc)
    end
    return s
end
function eval_scalar_with_grad(coeffs::BitMatrix, twoExps::Matrix{Int8}, as::Vector{TA}, t, n,S ) where {TA}
    s = zero(TA)
    coeff_vec = zeros(TA, t)
    for t_idx in 1:t
	 a = as[t_idx]
        iszero(a) && continue
        rows = get_term(t_idx, n)
        exp_acc = 0
        zero_hit = false
        @inbounds for m in 1:n
            r,c = rows[m], first(S[m])
            if !coeffs[r,c]
                zero_hit = true 
                break 
            end
            exp_acc += twoExps[r,c]
        end

        zero_hit && continue
	scale = ldexp(1.0, exp_acc)
        coeff_vec[t_idx] = scale
        s += a * scale
    end
    return s, coeff_vec
end
function _eval_broadcast(coeffs::BitMatrix, twoExps::Matrix{Int8}, as::Vector{TA}, t, n, S, nonscalar, dims::NTuple{K,Int}) where {K,TA}
    if K == 0
        s = zero(TA)
        for t_idx in 1:t
            a = as[t_idx]
            iszero(a) && continue
            rows = get_term(t_idx, n)
            exp_acc = 0
            zero_hit = false
            @inbounds for m in 1:n
                r,c = rows[m], first(S[m])
                if !coeffs[r,c]
                    zero_hit = true 
                    break 
                end
                exp_acc += twoExps[r,c]
            end
            zero_hit && continue
            s += ldexp(a,exp_acc)
        end
        return (s, s)
    end
    out = zeros(TA, dims)
    scalar_dims = [m for m in 1:n if length(S[m]) == 1]   # nur die "echten" Dimensionen
    for t_idx in 1:t
        a = as[t_idx];
        iszero(a) && continue
        rows = get_term(t_idx, n)
        coeff = a
        exp_acc = 0
        zero_hit = false
        @inbounds for m in scalar_dims
            r, c = rows[m], first(S[m])
            if !coeffs[r, c]
                zero_hit = true
                break
            end
            exp_acc += twoExps[r, c]
        end
        zero_hit && continue
        coeff = ldexp(a, exp_acc)
        @inbounds for k in 1:K
            mk = nonscalar[k]
            if !any(@view coeffs[rows[mk], S[mk]])
                zero_hit = true
                break
            end
        end
        zero_hit && continue
        exp_factors = ntuple(K) do k
            mk = nonscalar[k]
            sub = @view twoExps[rows[mk], S[mk]]
            shape = ntuple(d -> d == k ? dims[k] : 1, K)
            reshape(sub, shape...)
        end
        coeffs_factors = ntuple(K) do k
            mk = nonscalar[k]
            sub = @view coeffs[rows[mk], S[mk]]
            shape = ntuple(d -> d == k ? dims[k] : 1, K)
            reshape(sub, shape...)
        end
        out .+=  ifelse.(.&(coeffs_factors...),  ldexp.(coeff, .+(exp_factors...)), zero(TA))
    end
    return extrema(out)
end

function ChainRulesCore.rrule(::typeof(_eval_broadcast), bern_mat, as::Vector{TA}, t, n, S, nonscalar, dims::NTuple{K,Int}) where {K,TA}
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
            a = as[t_idx];
            iszero(a) && continue
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
function evaluate_reduced_tensor(coeffs, twoExps, as, t, n, S)
    lens = [  length(S[m]) for m in 1:n]
    nonscalar = [m for m in 1:n if lens[m] > 1]   # nur die "echten" Dimensionen


    dims = ntuple(k -> lens[nonscalar[k]], length(nonscalar))
    return _eval_broadcast(coeffs, twoExps, as, t, n, S, nonscalar,dims )

end
"""
compute min/ max for each term and add them all
"""
function imp_fast_bounds(as::AbstractArray, coeffs::BitMatrix, t::Int, n)
    b_min = zero(eltype(as))
    b_max = zero(eltype(as))

    # if orders == 2 each term looks like: [0,0,1] , [0, 0.5,1], [1,1,1]
    # if orders == 1 each term looks like: [0,1] , [1 , 1], 
    # therefore right_prod is always 1 and left_prod \in {0,1} <= right_prod 
    left_prods = @ignore_derivatives reshape(reduce(&, reshape(coeffs[:, 1], n, t), dims=1), t) # [left_prod_term1, left_prod_term2, ...  ]
    left_prod_with_coeff = as .* left_prods

    as_pos_mask = as .>= 0
    pos_part = ifelse.(as_pos_mask, as, zero(eltype(as)))
    neg_part = ifelse.(as_pos_mask,  zero(eltype(as)),as)

    lp_pos = ifelse.(as_pos_mask, left_prod_with_coeff, zero(eltype(as)))
    lp_neg = ifelse.(as_pos_mask,  zero(eltype(as)),left_prod_with_coeff)

    b_min = sum(lp_pos) + sum(neg_part)
    b_max = sum(lp_neg) + sum(pos_part)

    return b_min, b_max
end


function faster_exact_bounds(as::AbstractArray{N}, coeffs::BitMatrix, twoExps::Matrix{Int8}, order::Int8, n::Int, t::Int, constant_terms::BitArray, term_widths::AbstractArray{Int}, min_steps_j::TN; method=SmithBoundsOverapproximate, threshold=5) where {N<:Number,TN<:AbstractArray{N}}
    S_max = Vector{UnitRange{Int64}}(undef, n)
    S_min = Vector{UnitRange{Int64}}(undef, n)
    # expand bern_mat 

    non_constant_alphas = .!(isapprox.(as, 0; atol=1e-13))
    inc_mask = non_constant_alphas .&& as .> 0
    dec_mask = non_constant_alphas .&& .!inc_mask

    width_dec_full = sum(term_widths[dec_mask] .* abs.(as[dec_mask]))
    width_inc_full = sum(term_widths[inc_mask] .* abs.(as[inc_mask]))

    #diff_inc = sum(min_steps_j[increasing_mask,j] .* abs.(as[increasing_mask])  )

    all_diff_inc = min_steps_j' * (as .* inc_mask) # we can ignore constant terms here because min_steps_j is zero for them 

    all_diff_dec = min_steps_j' * (as .* dec_mask) # we can ignore constant terms here because min_steps_j is zero for them 

    @ignore_derivatives for x_i in 1:n
        S_min[x_i], S_max[x_i] = bound_algo(as, order, t, x_i, constant_terms, term_widths, non_constant_alphas, inc_mask, dec_mask, width_dec_full, width_inc_full, all_diff_dec, all_diff_inc)
    end
    min_possibilites = prod(length, S_min)
    if min_possibilites > threshold || min_possibilites < 0 # check for overflow
        if method == SmithBoundsOverapproximate
            b_min, b_max = imp_fast_bounds(as, coeffs, t, n)
        else
            b_min = -Inf
            b_max = Inf
        end
    elseif S_min == S_max
        #b_min, b_max = evaluate_reduced_tensor_cached(coeffs,twoExps,as,t,n,S_min)
        b_min, b_max = evaluate_reduced_tensor(coeffs, twoExps, as, t, n, S_min)
    else
        b_min, _ = evaluate_reduced_tensor(coeffs, twoExps, as, t, n, S_min)
        _, b_max = evaluate_reduced_tensor(coeffs, twoExps, as, t, n, S_max)
        #b_min, _ = evaluate_reduced_tensor_cached(coeffs, twoExps, as, t, n, S_min)
        #_, b_max = evaluate_reduced_tensor_cached(coeffs, twoExps, as, t, n, S_max)
    end

    return b_min, b_max
end

function get_required_dims(as::AbstractArray{N},  order::Int8, n::Int, t::Int, constant_terms::BitArray, term_widths::AbstractArray{Int}, min_steps_j::TN; method=SmithBoundsOverapproximate, threshold=5) where {N<:Number,TN<:AbstractArray{N}}
    S_max = Vector{UnitRange{Int64}}(undef, n)
    S_min = Vector{UnitRange{Int64}}(undef, n)
    # expand bern_mat 

    non_constant_alphas = .!(isapprox.(as, 0; atol=1e-13))
    inc_mask = non_constant_alphas .&& as .> 0
    dec_mask = non_constant_alphas .&& .!inc_mask

    width_dec_full = sum(term_widths[dec_mask] .* abs.(as[dec_mask]))
    width_inc_full = sum(term_widths[inc_mask] .* abs.(as[inc_mask]))

    #diff_inc = sum(min_steps_j[increasing_mask,j] .* abs.(as[increasing_mask])  )

    all_diff_inc = min_steps_j' * (as .* inc_mask) # we can ignore constant terms here because min_steps_j is zero for them 

    all_diff_dec = min_steps_j' * (abs.(as) .* dec_mask) # we can ignore constant terms here because min_steps_j is zero for them 

    @ignore_derivatives for x_i in 1:n
        S_min[x_i], S_max[x_i] = bound_algo(as, order, t, x_i, constant_terms, term_widths, non_constant_alphas, inc_mask, dec_mask, width_dec_full, width_inc_full, all_diff_dec, all_diff_inc)
    end

    min_possibilites = prod_capped(S_min; cap=threshold)

    if isnothing(min_possibilites)
        return nothing , nothing
    else 
        return S_min, S_max
    end

end

function precompute(coeffs::BitMatrix, twoExp::Matrix{Int8}, t, n, order)
    constant_terms = BitArray(undef, t, n)
    for t_idx in 1:t
        coeffs_term = @view coeffs[get_term(t_idx, n), :]
        twoExp_term = @view twoExp[get_term(t_idx, n), :]
        for var in 1:n
            constant_terms[t_idx, var] = all(coeffs_term[var, :]) && all(x -> x == 0, twoExp_term[var, :])
        end

    end
    term_widths = [1 - all(coeffs[get_term(t_idx, n), 1]) for t_idx in 1:t] # uses that the first element is always 0 or 1, and the last one is always on1

    #term_widths = [term_width((@view bern_mat[get_term(t_idx,n),:]) ,orders) for t_idx in 1:t ]

    min_steps_j = [term_min_step_along_j((@view coeffs[get_term(t_idx, n), :]), (@view twoExp[get_term(t_idx, n), :]), n, order, j)
                   for t_idx in 1:t, j in 1:n]
    return constant_terms, term_widths, min_steps_j
end

"""
Computes vector of lower bounds for Matrix of exponents.
"""
function monomial_lbs_0_1(E::M) where {M<:AbstractMatrix{<:Integer}}
    #one_odd = .~reduce(&, iseven.(E), dims = 1)
    # sum is only zero for a column if all rows are zero
    consts = sum(E, dims=1) .== 0

    # const columns have lb=1 (so we need to add 1 for them)
    # all other cols have lb=0 (so we don't need to add anything for them)
    #lbs = .- one_odd .+ consts
    return vec(consts)
end

"""
Computes interval bounds for each component of a sparse polynomial.
Variables are assumed to be in range [0, 1].
"""
function bounds_0_1(sp::SparsePolynomial{N,M,T,GM,EM,VI}) where {N,M,T,GM,EM,VI}

    n, m = size(sp.E)
    #lbs = [monomial_lb(ej) for ej in eachcol(sp.E)]
    lbs = @ignore_derivatives monomial_lbs_0_1(sp.E)
    ubs = ones(m)  # ub is just always 1 under our assumptions

    G⁻ = min.(zero(N), sp.G)
    G⁺ = max.(zero(N), sp.G)
    lb = G⁻ * ubs .+ G⁺ * lbs
    ub = G⁻ * lbs .+ G⁺ * ubs

    return lb, ub
end

function s_contains(outer::Vector{<:UnitRange}, inner::Vector{<:UnitRange})
    length(outer) == length(inner) || return false
    @inbounds for j in eachindex(outer)
        issubset(inner[j], outer[j]) || return false
    end
    return true
end
"""
returns indices of the maximum sets, owned with owned[i] = indices of all contained in Ss[i]
"""
function partition_maximal(Ss::Vector{Vector{UnitRange{Int64}}})
    m = length(Ss)
    is_dominated = falses(m)
    for i in 1:m
        for j in 1:m
            i == j && continue
            if Ss[i] != Ss[j] && s_contains(Ss[j], Ss[i])
                is_dominated[i] = true
                break
            end
        end
    end
    maximal_idx = findall(!, is_dominated)

    owned = [Int[] for _ in 1:length(maximal_idx)]
    for (j_idx, j) in enumerate(maximal_idx)
        for i in 1:m
            if s_contains(Ss[j],Ss[i])
                push!(owned[j_idx],i)
            end
        end
    end

    return maximal_idx, owned
end

function merge_S(a::Vector{UnitRange{Int}}, b::Vector{UnitRange{Int}})
    return [min(first(a[j]), first(b[j])):max(last(a[j]), last(b[j])) for j in eachindex(a)]
end
# avoids overflows
function prod_capped(dims; cap::Int)
    p = 1
    for d in dims
        wp = widemul(p, length(d))   # Int128, can't overflow here
        if wp > cap
            return nothing
        end
        p = Int(wp)                  # safe: wp <= cap, cap is an Int64
    end
    return p
end
function partition_maximal_merged_fast(Ss::Vector{Vector{UnitRange{Int}}}; threshold::Int)
    maximal_idx, owned = partition_maximal(Ss)

    cluster_S = Dict{Int, Vector{UnitRange{Int}}}()
    cluster_owned = Dict{Int, Vector{Int}}()
    next_id = 0
    for i in eachindex(maximal_idx)
        cluster_S[next_id] = Ss[maximal_idx[i]]
        cluster_owned[next_id] = copy(owned[i])
        next_id += 1
    end

    # heap entries: (size, id_a, id_b) -- lazily checked for staleness on pop
    heap = BinaryMinHeap{Tuple{Int,Int,Int}}()

    function push_candidates!(new_id::Int, ids)
        Snew = cluster_S[new_id]
        for other in ids
            other == new_id && continue
            cand = merge_S(Snew, cluster_S[other])
            sz = prod_capped(cand; cap=threshold)
            if sz !== nothing 
                a, b = min(new_id, other), max(new_id, other)
                push!(heap, (sz, a, b))
            end
        end
    end

    # seed heap with all initial pairs
    ids = collect(keys(cluster_S))
    for i in eachindex(ids)
        push_candidates!(ids[i], ids[i+1:end])
    end

    while !isempty(heap)
        sz, a, b = pop!(heap)

        # lazy deletion: skip if either cluster no longer exists
        (haskey(cluster_S, a) && haskey(cluster_S, b)) || continue

        Sa, Sb = cluster_S[a], cluster_S[b]
        merged = merge_S(Sa, Sb)

        @assert prod(length, merged) == sz

        new_owned = vcat(cluster_owned[a], cluster_owned[b])
        delete!(cluster_S, a); delete!(cluster_owned, a)
        delete!(cluster_S, b); delete!(cluster_owned, b)

        new_id = next_id; next_id += 1
        cluster_S[new_id] = merged
        cluster_owned[new_id] = new_owned

        push_candidates!(new_id, collect(keys(cluster_S)))
    end

    final_ids = collect(keys(cluster_S))
    return [cluster_S[id] for id in final_ids], [cluster_owned[id] for id in final_ids]
end


function get_coeffs(CMat, EMat, as::TN) where {TA <: Number, TN <: AbstractArray{TA}}
    total = size(EMat,2)
    out = zeros(TA, total)
    @inbounds for t_idx in  eachindex(as)
       a = as[t_idx]
       iszero(a) && continue 
       for j in 1:total 
            CMat[t_idx,j] || continue
            out[j] += ldexp(a,EMat[t_idx,j])
       end 
    end
    return out
end
function get_coeffs( Mfloat::Matrix{Float64}, as::TN) where {TA <: Number, TN <: AbstractArray{TA}}
    return  Mfloat' * as
end
function ChainRulesCore.rrule(::typeof(get_coeffs), Mfloat::Matrix{Float64}, as::Vector{TA}) where {TA<:Number}
    out = Mfloat' * as

    project_as = ProjectTo(as)

    function get_coeffs_pullback(Δout)
        Δout = unthunk(Δout)
        Δas = Mfloat * Δout
        return (NoTangent(), NoTangent(), project_as(Δas))
    end

    return out, get_coeffs_pullback
end



function bernstein_bounds(interval::CombinedPolyBernsteinInterval; method=Overapproximate, threshold=-1)
    num_p = size(interval.Low, 1)
    T = eltype(interval.Low)
    #lbs = Vector{T}(undef, num_p)
    #ubs = Vector{T}(undef, num_p)
    #for p_idx in 1:num_p
    #    lbs[p_idx], _ = imp_fast_bounds(interval.Low[p_idx, :], interval.bern_terms_coeffs, interval.t, interval.n)
    #    _, ubs[p_idx] = imp_fast_bounds(interval.Up[p_idx, :], interval.bern_terms_coeffs, interval.t, interval.n)


    #end

    if method == Overapproximate || SmithBoundsOverapproximate

	left_prods = @ignore_derivatives reshape(reduce(&, reshape(interval.bern_terms_coeffs[:, 1], interval.n, interval.t), dims=1), interval.t) # [left_prod_term1, left_prod_term2, ...  ]


	lbs = max.(interval.Low,0) * left_prods .+ min.(interval.Low,0) * ones(interval.t) 

	ubs = min.(interval.Up,0) * left_prods .+ max.(interval.Up,0) * ones(interval.t) 

	if method == Overapproximate
	    return lbs, ubs
	end
    else 
	lbs = fill(-Inf, num_p)
	ubs = fill(Inf, num_p)
    end
   
    constant_terms, term_widths, min_steps_j = precompute(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.t, interval.n, interval.order)
    needed_sets = Vector{Tuple{Vector{UnitRange{Int64}},Int,Symbol}}()
    for p_idx in 1:num_p
        S_min, _ = get_required_dims(interval.Low[p_idx, :], interval.order, interval.n, interval.t, constant_terms, term_widths, min_steps_j; method, threshold)
        _, S_max = get_required_dims(interval.Up[p_idx, :], interval.order, interval.n, interval.t, constant_terms, term_widths, min_steps_j; method, threshold)
        !isnothing(S_min) && push!(needed_sets, (S_min, p_idx, :low))
        !isnothing(S_max) && push!(needed_sets, (S_max, p_idx, :high))
    end

    maximal_sets, owned = partition_maximal_merged_fast(map(x -> x[1], needed_sets); threshold)

    for (i_idx, S) in enumerate(maximal_sets)
        lens = [length(S[m]) for m in 1:interval.n]
        nonscalar = [m for m in 1:interval.n if lens[m] > 1]
        scalar_dims = [m for m in 1:interval.n if lens[m] == 1]
        K = length(nonscalar)
        dims = ntuple(k -> lens[nonscalar[k]], Val(K))

        if K == 0
            for j in owned[i_idx]
                p_idx = needed_sets[j][2]
                if needed_sets[j][3] == :low
                    val = eval_scalar(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Low[p_idx,:], interval.t, interval.n, S)
                    lbs[p_idx] = max(lbs[p_idx], val)
                else
                    val = eval_scalar(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Up[p_idx,:], interval.t, interval.n, S)
                    ubs[p_idx] = min(ubs[p_idx], val)
                end
            end
        else
            Mfloat = build_M(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.t, interval.n, S, nonscalar, scalar_dims, dims)
            for j in owned[i_idx]
                p_idx = needed_sets[j][2]
                if needed_sets[j][3] == :low
                    lbs[p_idx] = max(lbs[p_idx], minimum(Mfloat' * interval.Low[p_idx,:]))
                else
                    ubs[p_idx] = min(ubs[p_idx], maximum(Mfloat' * interval.Up[p_idx,:]))
                end
            end
        end
    end

    return lbs, ubs
end

"""
Top-level bounds function: dispatches to bernstein_bounds for Overapproximate/SmithBoundsOverapproximate
(which has a custom, memory-safe rrule), or handles SmithBoundsMonomon by combining bernstein_bounds
with the monomial-basis bound -- this branch is NOT covered by a custom rrule, so Zygote traces it
normally (acceptable since it's not the branch causing memory issues).
"""
function bounds(interval::CombinedPolyBernsteinInterval; method=Overapproximate, threshold=-1)
    if method == Overapproximate || method == SmithBoundsOverapproximate
        return bernstein_bounds(interval; method, threshold)
    elseif method == SmithBoundsMonomon
        lbs, ubs = bernstein_bounds(interval; method=SmithBoundsOverapproximate, threshold)
        poly_interval = to_sparse_polynomial(interval)
        splbs, _ = bounds(poly_interval.Low)
        _, spubs = bounds(poly_interval.Up)
        return max.(lbs, splbs), min.(ubs, spubs)
    else
        error("unknown method $method")
    end
end

function ChainRulesCore.rrule(::typeof(bernstein_bounds), interval::CombinedPolyBernsteinInterval; method=Overapproximate, threshold=-1)
    num_p = size(interval.Low, 1)
    T = eltype(interval.Low)

    lbs = Vector{T}(undef, num_p)
    ubs = Vector{T}(undef, num_p)
    winning_low_coeffs = Vector{Vector{T}}(undef, num_p)
    winning_up_coeffs  = Vector{Vector{T}}(undef, num_p)

    left_prods = reshape(reduce(&, reshape(interval.bern_terms_coeffs[:, 1], interval.n, interval.t), dims=1), interval.t)

    for p_idx in 1:num_p
        as_low = interval.Low[p_idx, :]
        as_up  = interval.Up[p_idx, :]

        lo, _ = imp_fast_bounds(as_low, interval.bern_terms_coeffs, interval.t, interval.n)
        _, hi = imp_fast_bounds(as_up,  interval.bern_terms_coeffs, interval.t, interval.n)
        lbs[p_idx] = lo
        ubs[p_idx] = hi

        # d(b_min)/d(as[i]) = left_prods[i] if as[i]>=0 else 1
        # d(b_max)/d(as[i]) = 1 if as[i]>=0 else left_prods[i]
        winning_low_coeffs[p_idx] = ifelse.(as_low .>= 0, left_prods, one(T))
        winning_up_coeffs[p_idx]  = ifelse.(as_up  .>= 0, one(T), left_prods)
    end

    if method == Overapproximate
        function pullback_overapprox(Δ)
            Δlbs, Δubs = unthunk(Δ)
            dLow = zeros(T, num_p, interval.t)
            dUp  = zeros(T, num_p, interval.t)
            for p_idx in 1:num_p
                dLow[p_idx, :] .= Δlbs[p_idx] .* winning_low_coeffs[p_idx]
                dUp[p_idx, :]  .= Δubs[p_idx] .* winning_up_coeffs[p_idx]
            end
            dinterval = Tangent{CombinedPolyBernsteinInterval}(Low=dLow, Up=dUp)
            return (NoTangent(), dinterval)
        end
        return (lbs, ubs), pullback_overapprox
    end

    # method == SmithBoundsOverapproximate: refine via clusters
    constant_terms, term_widths, min_steps_j = precompute(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.t, interval.n, interval.order)
    needed_sets = Vector{Tuple{Vector{UnitRange{Int64}},Int,Symbol}}()
    for p_idx in 1:num_p
        S_min, _ = get_required_dims(interval.Low[p_idx, :], interval.order, interval.n, interval.t, constant_terms, term_widths, min_steps_j; method, threshold)
        _, S_max = get_required_dims(interval.Up[p_idx, :], interval.order, interval.n, interval.t, constant_terms, term_widths, min_steps_j; method, threshold)
        !isnothing(S_min) && push!(needed_sets, (S_min, p_idx, :low))
        !isnothing(S_max) && push!(needed_sets, (S_max, p_idx, :high))
    end

    maximal_sets, owned = partition_maximal_merged_fast(map(x -> x[1], needed_sets); threshold)

    for (i_idx, S) in enumerate(maximal_sets)
        lens = [length(S[m]) for m in 1:interval.n]
        nonscalar = [m for m in 1:interval.n if lens[m] > 1]
        scalar_dims = [m for m in 1:interval.n if lens[m] == 1]
        K = length(nonscalar)
        dims = ntuple(k -> lens[nonscalar[k]], Val(K))

        if K == 0
            for j in owned[i_idx]
                p_idx = needed_sets[j][2]
                if needed_sets[j][3] == :low
                    val, coeff_vec = eval_scalar_with_grad(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Low[p_idx,:], interval.t, interval.n, S)
                    if val > lbs[p_idx]
                        lbs[p_idx] = val
                        winning_low_coeffs[p_idx] = coeff_vec
                    end
                else
                    val, coeff_vec = eval_scalar_with_grad(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Up[p_idx,:], interval.t, interval.n, S)
                    if val < ubs[p_idx]
                        ubs[p_idx] = val
                        winning_up_coeffs[p_idx] = coeff_vec
                    end
                end
            end
        else
            Mfloat = build_M(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.t, interval.n, S, nonscalar, scalar_dims, dims)
            for j in owned[i_idx]
                p_idx = needed_sets[j][2]
                if needed_sets[j][3] == :low
                    vals = Mfloat' * interval.Low[p_idx,:]
                    val, flat_idx = findmin(vals)
                    if val > lbs[p_idx]
                        lbs[p_idx] = val
                        winning_low_coeffs[p_idx] = copy(@view Mfloat[:, flat_idx])
                    end
                else
                    vals = Mfloat' * interval.Up[p_idx,:]
                    val, flat_idx = findmax(vals)
                    if val < ubs[p_idx]
                        ubs[p_idx] = val
                        winning_up_coeffs[p_idx] = copy(@view Mfloat[:, flat_idx])
                    end
                end
            end
        end
    end

    function pullback(Δ)
        Δlbs, Δubs = unthunk(Δ)
        dLow = zeros(T, num_p, interval.t)
        dUp  = zeros(T, num_p, interval.t)
        for p_idx in 1:num_p
            dLow[p_idx, :] .= Δlbs[p_idx] .* winning_low_coeffs[p_idx]
            dUp[p_idx, :]  .= Δubs[p_idx] .* winning_up_coeffs[p_idx]
        end
        dinterval = Tangent{CombinedPolyBernsteinInterval}(Low=dLow, Up=dUp)
        return (NoTangent(), dinterval)
    end

    return (lbs, ubs), pullback
end



function all_bounds(interval::CombinedPolyBernsteinInterval; method=Overapproximate, threshold=-1)
    num_p = size(interval.Low, 1)
    T = eltype(interval.Low)
    llbs = Vector{T}()
    lubs = Vector{T}()
    ulbs = Vector{T}()
    uubs = Vector{T}()
    sizehint!(llbs, num_p)
    sizehint!(lubs, num_p)
    sizehint!(ulbs, num_p)
    sizehint!(uubs, num_p)
    empty!(M_CACHE)

    if method == SmithBoundsMonomon
        poly_interval = to_sparse_polynomial(interval)
        llbs_sp, lubs_sp = bounds(poly_interval.Low)
        ulbs_sp, uubs_sp = bounds(poly_interval.Up)

    end
    function round_key(row, digits=10)
        return round.(row; digits=digits)  # returns a Vector, hashable
    end
    function find_or_insert!(unique_as, buckets, row, entry)
        key = round_key(row)
        candidates = get!(buckets, key, Int[])
        for c in candidates
            p_idx, is_low = unique_as[c]
            ref = is_low ? (@view interval.Low[p_idx, :]) : (@view interval.Up[p_idx, :])
            if isapprox(row, ref)
                return c
            end
        end
        push!(unique_as, entry)
        push!(candidates, length(unique_as))
        return length(unique_as)
    end

    unique_as = Vector{Tuple{Int64,Bool}}()

    buckets = Dict{Vector{Float64}, Vector{Int}}()  # rounded-key -> indices into unique_as

    #low_evaluated_poly = Array{Int64}(undef, size(interval.Low, 1))
    #@ignore_derivatives for i in axes(interval.Low, 1)
    #    idx = findfirst(x -> isapprox((@view interval.Low[i, :]), (@view interval.Low[x[1], :])), unique_as)
    #    if isnothing(idx)
    #        push!(unique_as, (i, true))
    #        low_evaluated_poly[i] = size(unique_as, 1)
    #    else
    #        low_evaluated_poly[i] = idx
    #    end
    #end

    #up_evaluated_poly = Array{Int64}(undef, size(interval.Up, 1))
    #@ignore_derivatives for i in axes(interval.Up, 1)
    #    idx = findfirst(x -> isapprox((@view interval.Up[i, :]), (x[2] ? (@view interval.Low[x[1], :]) : @view interval.Up[x[1], :])), unique_as)
    #    if isnothing(idx)
    #        push!(unique_as, (i, false))
    #        up_evaluated_poly[i] = size(unique_as, 1)
    #    else
    #        up_evaluated_poly[i] = idx
    #    end
    #end
    # reenable after here
    low_evaluated_poly = Array{Int64}(undef, size(interval.Low, 1))
    @ignore_derivatives for i in axes(interval.Low, 1)
        row = @view interval.Low[i, :]
        low_evaluated_poly[i] = find_or_insert!(unique_as, buckets, row, (i, true))
    end

    up_evaluated_poly = Array{Int64}(undef, size(interval.Up, 1))
    @ignore_derivatives for i in axes(interval.Up, 1)
        row = @view interval.Up[i, :]
        up_evaluated_poly[i] = find_or_insert!(unique_as, buckets, row, (i, false))
    end
    println("saved $(2*num_p - length(unique_as)) polynomials")

    if method != Overapproximate
        constant_terms, term_widths, min_steps_j = @ignore_derivatives precompute(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.t, interval.n, interval.order)
    end
    unique_bounds = Zygote.Buffer(Array{Tuple{Float64,Float64}}(undef, 1), size(interval.Low, 1))
    if method == Overapproximate
        for i in axes(interval.Low,1)
            p_idx, is_lower = unique_as[i]
            if is_lower
                unique_bounds[i] = imp_fast_bounds(interval.Low[p_idx, :], interval.bern_terms_coeffs, interval.t, interval.n)
            else
                unique_bounds[i] = imp_fast_bounds(interval.Up[p_idx, :], interval.bern_terms_coeffs, interval.t, interval.n)
            end
        end
    end
    if method != Overapproximate
    # in  a first pass calculate the required dimensions.
        needed_sets = Vector{Tuple{Vector{UnitRange},Int,Symbol}}()
        use_approx = Int[]
        for i in eachindex(unique_as)
            p_idx, is_lower = unique_as[i]
            if is_lower
                S_min, S_max =  get_required_dims(interval.Low[p_idx, :],interval.order, interval.n,interval.t,constant_terms,term_widths,min_steps_j; method,threshold )
            else
                S_min, S_max =  get_required_dims(interval.Up[p_idx, :],interval.order, interval.n,interval.t,constant_terms,term_widths,min_steps_j; method,threshold )
            end
            if isnothing(S_min)
                push!(use_approx,i)
            elseif S_min == S_max 
                push!(needed_sets,(S_min,i,:both))
            else
                push!(needed_sets,(S_min,i,:low))
                push!(needed_sets,(S_max,i,:high))
            end

        end
        maximal_sets, owned =  partition_maximal(map(x -> x[1], needed_sets ))
	println( "saved $(length(needed_sets) -  length(maximal_sets)) sets") 
        for (i_idx,i) in enumerate(maximal_sets)
            S = needed_sets[i][1]

            lens = [length( S[m]) for m in 1:interval.n]
            nonscalar = [m for m in  1:interval.n if lens[m] > 1]
            scalar_dims = [m for m in 1:interval.n if lens[m] == 1]
            K = length(nonscalar)
            dims = ntuple(k -> lens[nonscalar[k]], Val(K))
            


            if K != 0
                CMat, EMat = build_M( interval.bern_terms_coeffs,interval.bern_terms_2_exp,interval.t,interval.n,S,nonscalar,scalar_dims,dims)
            end

            for j in owned[i_idx]
                unique_as_idx = needed_sets[j][2]
                p_idx, is_lower= unique_as[unique_as_idx]
                
                cur_unique_bounds = unique_bounds[unique_as_idx]
                if is_lower 
                   if needed_sets[j][3] == :low
                        unique_bounds[unique_as_idx] = (K == 0 ?
                         _eval_broadcast(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Low[p_idx,:], interval.t, interval.n, S, nonscalar, dims)[1] :
                          minimum(get_coeffs(CMat,EMat,interval.Low[p_idx,:]))  , cur_unique_bounds[2])
                   elseif needed_sets[j][3] == :high
                        unique_bounds[unique_as_idx] = (cur_unique_bounds[1],K == 0 ?
                         _eval_broadcast(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Low[p_idx,:], interval.t, interval.n, S, nonscalar, dims)[2] :
                          maximum(get_coeffs(CMat,EMat,interval.Low[p_idx,:])) )
                   else 
                        unique_bounds[unique_as_idx] = K == 0 ?
                         _eval_broadcast(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Low[p_idx,:], interval.t, interval.n, S, nonscalar, dims) :
                          extrema(get_coeffs(CMat,EMat,interval.Low[p_idx,:])) 
                   end
                else
                   if needed_sets[j][3] == :low
                        unique_bounds[unique_as_idx] = (K == 0 ?
                         _eval_broadcast(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Up[p_idx,:], interval.t, interval.n, S, nonscalar, dims)[1] :
                          minimum(get_coeffs(CMat,EMat,interval.Up[p_idx,:])),cur_unique_bounds[2])
                   elseif needed_sets[j][3] == :high  
                        unique_bounds[unique_as_idx] = (cur_unique_bounds[1],K == 0 ?
                         _eval_broadcast(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Up[p_idx,:], interval.t, interval.n, S, nonscalar, dims)[2] :
                          maximum(get_coeffs(CMat,EMat,interval.Up[p_idx,:])))
                   else 
                        unique_bounds[unique_as_idx] = K == 0 ?
                         _eval_broadcast(interval.bern_terms_coeffs, interval.bern_terms_2_exp, interval.Up[p_idx,:], interval.t, interval.n, S, nonscalar, dims) :
                          extrema(get_coeffs(CMat,EMat,interval.Up[p_idx,:]))
                   end
                end
            end
        end
        for i in use_approx
           p_idx, is_lower = unique_as[i] 
           if is_lower
                if method == SmithBoundsOverapproximate
                    unique_bounds[i] = imp_fast_bounds(interval.Low[p_idx, :], interval.bern_terms_coeffs, interval.t, interval.n)
                else
                    unique_bounds[i] = -Inf, Inf
                end
            else
                if method == SmithBoundsOverapproximate
                    unique_bounds[i] = imp_fast_bounds(interval.Up[p_idx, :], interval.bern_terms_coeffs, interval.t, interval.n)
                else
                    unique_bounds[i] = -Inf, Inf
                end
            end
        end
    end




    for p_idx in axes(interval.Low, 1)

        llbsi, lubsi = unique_bounds[low_evaluated_poly[p_idx]]
        ulbsi, uubsi = unique_bounds[up_evaluated_poly[p_idx]]

#        llbsF, lubsF = faster_exact_bounds(interval.Low[p_idx,:],interval.bern_terms_coeffs,interval.bern_terms_2_exp,interval.order,interval.n,interval.t,constant_terms,term_widths,min_steps_j;threshold,method)

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
    return llbs, lubs, ulbs, uubs

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
    return bounds(mapped_interval; method, threshold)
end

function get_term(t_idx, n)
    return ((t_idx-1)*n+1):(t_idx*n)
end
"""
remove duplicate terms
"""
function combine_terms(inter::CombinedPolyBernsteinInterval)
    unique_term_coeffs = BitMatrix[]
    unique_term_twoExps = Matrix{Int8}[]
    Low = zeros(eltype(inter.Low), size(inter.Up, 1), 0)   # num_polys×0 matrix
    Up = zeros(eltype(inter.Up), size(inter.Up, 1), 0)   # num_polys×0 matrix
    for old_term_idx in 1:inter.t
        term_coeffs = inter.bern_terms_coeffs[get_term(old_term_idx, inter.n), :]
        term_twoExps = inter.bern_terms_2_exp[get_term(old_term_idx, inter.n), :]
        t_idx = findfirst(x -> x == term_coeffs, unique_term_coeffs)
        if isnothing(t_idx) || term_twoExps[term_coeffs] != unique_term_twoExps[t_idx][unique_term_coeffs[t_idx]]
            unique_term_coeffs = [unique_term_coeffs..., copy(term_coeffs)]
            unique_term_twoExps = [unique_term_twoExps..., copy(term_twoExps)]
            Low = [Low zeros(eltype(Low), size(Low, 1))]
            Up = [Up zeros(eltype(Up), size(Up, 1))]
            t_idx = length(unique_term_coeffs)
        end
        #Low[:,t_idx] .+=  inter.Low[:, old_term_idx]
        Low = hcat(Low[:, 1:(t_idx-1)],
            Low[:, t_idx] .+ inter.Low[:, old_term_idx],
            Low[:, (t_idx+1):end])
        Up = hcat(Up[:, 1:(t_idx-1)],
            Up[:, t_idx] .+ inter.Up[:, old_term_idx],
            Up[:, (t_idx+1):end])
        #Up[:,t_idx] .+=  inter.Up[:, old_term_idx]
    end
    return CombinedPolyBernsteinInterval(Low, Up, vcat(unique_term_coeffs...), vcat(unique_term_twoExps...), length(unique_term_coeffs), inter.n, inter.order, inter.X, inter.lbs, inter.ubs)
    # look for zero columns: 
    #new_Low =  zeros(eltype(inter.Low), size(inter.Low,1), 0)   # num_polys×0 matrix
    #new_Up =  zeros(eltype(inter.Up), size(inter.Up,1), 0)   # num_polys×0 matrix
    #new_unique_terms  = Matrix{eltype(inter.bern_terms)}[]
    #for col_idx in axes(Low,2)
    #    low_col = @view Low[:,col_idx]
    #    up_col = @view Up[:,col_idx]
    #    if all(low_col .== 0) && all(up_col .== 0)
    #        continue
    #    end
    #    new_unique_terms = [new_unique_terms... ,unique_terms[col_idx]]
    #    new_Low = [new_Low low_col]
    #    new_Up = [new_Up up_col]
    #    
    #end
    #if isempty(new_unique_terms)
    #    new_unique_terms = [new_unique_terms..., constant_term(eltype(inter.bern_terms),inter.n,inter.orders)]
    #    new_Low =  zeros(eltype(inter.Low), size(inter.Up,1), 1)   # num_polys×1 matrix
    #    new_Up =  zeros(eltype(inter.Up), size(inter.Up,1), 1)   # num_polys×1 matrix
    #end
    #return CombinedPolyBernsteinInterval(new_Low,new_Up,vcat(new_unique_terms...),length(new_unique_terms),inter.n,inter.orders,inter.X,inter.lbs,inter.ubs)

end
