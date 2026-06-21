using LazySets
using JLD2;
struct MultiBernsteinImp{M<:Integer,O<:Number,TN<:AbstractArray{O}}
    coefficient_matrix::TN # matrix containing coefficients 
    t::Vector{M} # number of terms
    orders::Vector{Vector{Int64}} # orders
end
struct BernsteinInterval{N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}, VN<:AbstractArray{N}}
    Low::MultiBernsteinImp{M,O,TN}
    Up::MultiBernsteinImp{M,O,TN}
    n::M # input dimension
    X::Hyperrectangle{N}
    lbs::Vector{VN}  # (vector of vectors) lower bounds for intermediate values
    ubs::Vector{VN}  # (vector of vectors) upper bounds for intermediate values
end

function BernsteinInterval(Low::MultiBernsteinImp{M,O,TN}, Up::MultiBernsteinImp{M,O,TN}, n::M, X::Hyperrectangle{N}) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    return BernsteinInterval(Low, Up, n,X,Vector{Vector{N}}(), Vector{Vector{N}}())
end

"""
Construct a MultiBernsteinImp from a list of BernsteinPolynomialImp 

"""
function build_multi(bern_polys::Vector{<:BernsteinPolynomialImp})

    res = zeros(
        sum(map(x -> size(x.coefficient_matrix, 1), bern_polys)),
	maximum(map(x -> maximum(x.orders), bern_polys))+1,
    )
    res_offset = 1
    orders = similar(Vector{Vector{Int64}}, length(bern_polys))
    t = similar(Vector{typeof(bern_polys[1].t)}, length(bern_polys))
    for (i, poly) in enumerate(bern_polys)
        res[res_offset:(res_offset+size(poly.coefficient_matrix, 1)-1), 1: size(poly.coefficient_matrix,2) ] =
            poly.coefficient_matrix
        res_offset += size(poly.coefficient_matrix, 1)
        orders[i] = poly.orders
        t[i] = poly.t
    end
    return MultiBernsteinImp(res, t, orders)
end

"""
Construct a bernstein Polynomial repr of x_i over the input domain X 
"""
function init_one_coeff_mat(poly_idx,num_poly_vars, input_idx, X::Hyperrectangle)
    coeff_mat = fill(1.0, num_poly_vars, 2)
    coeff_mat[poly_idx, :] = [low(X)[input_idx], high(X)[input_idx]]
    return coeff_mat
end

"""
Construct a BernsteinInterval over h by filling Low and Up with x for the unfixed variables
"""
function init_bernstein_interval(h::Hyperrectangle)
    num_polys = dim(h)
    unfixed_mask = (h.radius .!= 0)
    n = count(unfixed_mask)
    #@show unfixed_mask,n
    @polyvar x[1:num_polys]
    coeff_matrices = similar(h.radius, n*num_polys ,2  )
    
    for i in 1:num_polys
	    coeff_idx = (i-1)*n +1
	    if unfixed_mask[i]
	        coeff_matrices[coeff_idx:coeff_idx + n - 1, :]  = init_one_coeff_mat(count(@view unfixed_mask[1:i]),n,i,h)
	    else 
	        # constant value
	        coeff_mat = fill(1.0, n, 2)
	        coeff_mat[1,:] .= low(h)[i]
	        coeff_matrices[coeff_idx:coeff_idx + n - 1, :] = coeff_mat
	    end
    end
    return BernsteinInterval(
        MultiBernsteinImp(coeff_matrices, fill(1, num_polys), fill(fill(1, n), num_polys)),
        MultiBernsteinImp(copy(coeff_matrices), fill(1, num_polys), fill(fill(1, n), num_polys)),
        num_polys,
        h,
    )
end

"""
checks for terms that are all zeros and gets rid of them  
"""
function remove_zero_terms(multi::MultiBernsteinImp)
    remaining_terms = deepcopy(multi.t)
    n = length(multi.orders[1])
    new_coeffs = []
    offset =1 
    for poly_idx in eachindex(remaining_terms)
        poly_offset = 0
        kept_a_term = false
        #println("first while condition: $(remaining_terms[poly_idx] > 1 )and $(poly_offset < n*(multi.t[poly_idx]-1))")
        while(remaining_terms[poly_idx] > 1 && poly_offset <= n*(multi.t[poly_idx]-1)) # keep zero if it is the only term
            zero_row = false 
            #@show poly_offset, poly_idx,multi.t[poly_idx],n
            for var_idx in 0:(n-1)
                if all(==(0),multi.coefficient_matrix[offset+poly_offset+var_idx , : ]) 
                    zero_row = true 
                    break 
                end
            end
            if zero_row 
                #@show poly_idx,poly_offset
                #@show multi.coefficient_matrix[offset+poly_offset:(offset + poly_offset +  n-1),: ]
                remaining_terms[poly_idx]-=1
                #println("throwing out:")
            else 
                #println("keeping: ")
                kept_a_term = true
                push!(new_coeffs, multi.coefficient_matrix[offset+poly_offset:(offset + poly_offset +  n-1),: ])
                #@show new_coeffs
            end
            #@show poly_offset
            #println("adding!")
            poly_offset += n
            #@show poly_offset
        end
        if (!kept_a_term)
            push!(new_coeffs, multi.coefficient_matrix[offset+poly_offset:(offset + poly_offset +  n-1),: ])
            poly_offset += n
        end
        offset += poly_offset
    end
    #@show new_coeffs 
    #@show remaining_terms 
    return MultiBernsteinImp(reduce(vcat,new_coeffs),remaining_terms,copy(multi.orders))
end
function linear_map(
    mat,
    polys::Vector{BernsteinPolynomialImp{N,M,O,TN}},
) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    res_polys = similar(polys, size(mat, 1))
    for (i, row) in enumerate(eachrow(mat))
        #println("row: $row")
        #println("polys: $polys")
        lin_polys = [scalar_mul(poly, c) for (c, poly) in zip(row, polys)]
        #println("lin_polys: $lin_polys")
        res_polys[i] = foldl(add, lin_polys)
    end
    return res_polys
end

""" 
calculate A * multi
 assumes all polys in multi have the same orders!
"""
function linear_map(
    A,
    multi::MultiBernsteinImp{M,O,TN};
    use_memory_optimizations=true
) where {M<:Integer,O<:Number,TN<:AbstractArray{O}}
    final_coeffs = similar(
        multi.coefficient_matrix,
        (size(A, 1)*size(multi.coefficient_matrix, 1), size(multi.coefficient_matrix, 2)),
    )
    for (i, row) in enumerate(eachrow(A))
        start_idx = (i-1)*size(multi.coefficient_matrix, 1) + 1
        end_idx = (i)*size(multi.coefficient_matrix, 1)
        final_coeffs[start_idx:end_idx, :] .= scalar_mul(multi, row)
    end
    newt = fill(sum(multi.t), size(A, 1))
    if use_memory_optimizations
        return combine_terms(MultiBernsteinImp(final_coeffs, newt, fill(multi.orders[1], size(A, 1))))
    end
    return MultiBernsteinImp(final_coeffs, newt, fill(multi.orders[1], size(A, 1)))
end

"""
elevate orders of all contained polynomials to degrees ,
"""
function elevate_all_to(multi::MultiBernsteinImp, new_orders::Vector{Int64};use_memory_optimizations=true)
    @assert all(sub -> all(sub .<= new_orders), multi.orders)
    diffs = [new_orders - cur_orders for cur_orders in multi.orders]
    res = similar(multi.coefficient_matrix)
    start = 1
    for i = 1:length(multi.orders)
        end_idx = start+(multi.t[i]*length(multi.orders[1])) - 1
        #@show start, end_idx
        #@show repeat(binomial_tensor(new_orders),multi.t[i],1)
        res[start:end_idx, :] =
            multi.coefficient_matrix[start:end_idx, :] .*
            repeat(binomial_tensor(multi.orders[i]), multi.t[i], 1)
        start += multi.t[i]*length(multi.orders[1])
    end
    max_diff = maximum(maximum.(diffs))
    C_diff = zeros(size(multi.coefficient_matrix, 1), max_diff+1)
    start = 1
    for i in eachindex(diffs)
        repeated = repeat(binomial_tensor(diffs[i]), multi.t[i], 1)
        C_diff[start:(start+(multi.t[i]*length(multi.orders[1]))-1), 1:size(repeated, 2)] =
            repeated
        start += multi.t[i]*length(multi.orders[1])
    end

    C_new_orders = repeat(binomial_tensor(new_orders), sum(multi.t), 1)

    result = row_convolution_kernel(res, C_diff)
    result = result[:, 1:(maximum(new_orders)+1)]
    result = ifelse.(C_new_orders .!= 0, result ./ C_new_orders, 0.0)
    if use_memory_optimizations
        return combine_terms(MultiBernsteinImp(result, multi.t, fill(new_orders, length(multi.orders))))
    end
    return MultiBernsteinImp(result, multi.t, fill(new_orders, length(multi.orders)))
end



"""
elevate orders of all contained polynomials to degrees ,
"""
function elevate(multi::MultiBernsteinImp, new_orders::Vector{Vector{Int64}};use_memory_optimizations=true)
    #@assert all(sub -> all(sub .<= new_orders), multi.orders)
    @assert length(new_orders) == length(multi.orders)
    diffs = [new_orders[i] - multi.orders[i] for i in eachindex(multi.orders)]
    res = zeros(size(multi.coefficient_matrix))
    start = 1


    for i = 1:length(multi.orders)
        end_idx = start+(multi.t[i]*length(multi.orders[1])) - 1
        #@show start, end_idx
        #@show repeat(binomial_tensor(new_orders),multi.t[i],1)
        res[start:end_idx, 1:maximum(multi.orders[i])+1] =
            multi.coefficient_matrix[start:end_idx, 1: maximum(multi.orders[i])+1 ] .*
            repeat(binomial_tensor(multi.orders[i]), multi.t[i], 1)
        start += multi.t[i]*length(multi.orders[1])
    end
    max_diff = maximum(maximum.(diffs))
    C_diff = zeros(size(multi.coefficient_matrix, 1), max_diff+1)
    start = 1
    for i in eachindex(diffs)
        repeated = repeat(binomial_tensor(diffs[i]), multi.t[i], 1)
        C_diff[start:(start+(multi.t[i]*length(multi.orders[1]))-1), 1:size(repeated, 2)] =
            repeated
        start += multi.t[i]*length(multi.orders[1])
    end

    C_new_orders = zeros(size(multi.coefficient_matrix, 1), maximum(maximum.(new_orders))+1 )
    start = 1
    for i in eachindex(new_orders) 
	    repeated = repeat(binomial_tensor(new_orders[i]),multi.t[i],1)
	    C_new_orders[start:(start+(multi.t[i]*length(multi.orders[1]))-1), 1:size(repeated,2)] = repeated
	    start += multi.t[i]*length(multi.orders[1])
    end

    result = row_convolution_kernel(res, C_diff)
    result = result[:, 1:(maximum(maximum.(new_orders))+1)]
    result = ifelse.(C_new_orders .!= 0, result ./ C_new_orders, 0.0)
    if use_memory_optimizations
        return  combine_terms(MultiBernsteinImp(result, multi.t, new_orders ))
    end
    return MultiBernsteinImp(result,multi.t,new_orders)
end

function add(
    multi_a::MultiBernsteinImp{M,O,TN},
    multi_b::MultiBernsteinImp{M,O,TN};
    use_memory_optimizations=true
) where {M<:Integer,O<:Number,TN<:AbstractArray{O}}
    @assert length(multi_a.t) == length(multi_b.t)
    new_orders = [
        max.(order_a, order_b) for (order_a, order_b) in zip(multi_a.orders, multi_b.orders)
    ]
    #@show new_orders
    new_order = maximum(new_orders)
    #@show new_order
    n = length(new_order)
    elevated_a = elevate(multi_a, new_orders)
    elevated_b = elevate(multi_b, new_orders)
    rows_a, cols_a = size(elevated_a.coefficient_matrix)
    rows_b, _ = size(elevated_b.coefficient_matrix)
    res = similar(elevated_a.coefficient_matrix, rows_a+rows_b, cols_a)
    res_offset = 1
    a_offset = 1
    b_offset = 1
    new_t = similar(elevated_a.t)
    for i in eachindex(elevated_a.t)
        # get one polynom from a 
        add_a_offset = (elevated_a.t[i]*n) - 1
        res[res_offset:(res_offset+add_a_offset), :] =
            elevated_a.coefficient_matrix[a_offset:(a_offset+add_a_offset), :]
        res_offset += add_a_offset + 1
        a_offset += add_a_offset + 1
        # and one polynom from b
        add_b_offset = (elevated_b.t[i]*n) - 1
        res[res_offset:(res_offset+add_b_offset), :] =
            elevated_b.coefficient_matrix[b_offset:(b_offset+add_b_offset), :]
        res_offset += add_b_offset + 1
        b_offset += add_b_offset + 1
        new_t[i] = elevated_a.t[i] + elevated_b.t[i]
    end
    if use_memory_optimizations
        return combine_terms(MultiBernsteinImp(res,new_t,new_orders))
    end
    return MultiBernsteinImp(res, new_t, new_orders)

end

function add(
    polys_a::Vector{<:BernsteinPolynomialImp},
    polys_b::Vector{<:BernsteinPolynomialImp},
)
    return [add(poly_a, poly_b) for (poly_a, poly_b) in zip(polys_a, polys_b)]
end

function translate(polys::Vector{<:BernsteinPolynomialImp}, bs::Vector{N}) where {N<:Number}
    return [translate(poly, b) for (poly, b) in zip(polys, bs)]
end
function translate(multi::MultiBernsteinImp, bs::Vector{N}) where {N<:Number}
    if all(==(0),bs) 
        return multi
    end
    @assert length(bs) == length(multi.t)
    res = similar(
        multi.coefficient_matrix,
        size(multi.coefficient_matrix, 1)+length(bs)*length(multi.orders[1]),
        size(multi.coefficient_matrix, 2),
    )
    res_offset = 1
    multi_offset = 1
    n = length(multi.orders[1])
    for i in eachindex(multi.t)
        add_multi_offset = (multi.t[i]*n) - 1
        res[res_offset:(res_offset+add_multi_offset), :] =
            multi.coefficient_matrix[multi_offset:(multi_offset+add_multi_offset), :]
        res_offset += add_multi_offset + 1
        multi_offset += add_multi_offset + 1
        b_coeff_mat = fill(0.0, n, size(res, 2))
        for (i, order) in enumerate(multi.orders[i])
            b_coeff_mat[i, 1:(order+1)] .= 1
        end
        b_coeff_mat[1, :] .*= bs[i]
        res[res_offset:(res_offset+n-1), :] = b_coeff_mat
        res_offset += n
    end
    return MultiBernsteinImp(res, multi.t .+ 1, multi.orders)
end

"""
Interval map overapproximating an affine map Wx + b for x ∈ I 

W⁻ - (matrix) negative weights
W⁺ - (matrix) positive weights
I  - (BernsteinInterval) bernsteinInterval
b  - (vector) bias
"""
function interval_map(W⁻, W⁺, I::BernsteinInterval, b; use_memory_optimizations=true)
    #@show W⁻ , W⁺
    new_low = add(linear_map(W⁻, I.Up;use_memory_optimizations), linear_map(W⁺, I.Low;use_memory_optimizations)) 
    new_up = add(linear_map(W⁻, I.Low;use_memory_optimizations), linear_map(W⁺, I.Up;use_memory_optimizations)) 
    new_low = translate(new_low, b)
    new_up = translate(new_up, b)
    return BernsteinInterval(new_low, new_up, I.n, I.X)
end
function interval_maplists(
    W⁻,
    W⁺,
    Low::Vector{<:BernsteinPolynomialImp},
    Up::Vector{<:BernsteinPolynomialImp},
    b,
)
    new_low = add(linear_map(W⁻, Up), linear_map(W⁺, Low))
    new_up = add(linear_map(W⁻, Low), linear_map(W⁺, Up))
    new_low = translate(new_low, b)
    new_up = translate(new_up, b)
    return (new_low, new_up)
end

function scalar_mul(multi::MultiBernsteinImp, n::Integer, scalar)::MultiBernsteinImp
    C = copy(multi.coefficient_matrix)

    rows, cols = size(C)

    for i = 1:n:rows
        C[i, :] .*= scalar
    end

    return MultiBernsteinImp(C, multi.t, multi.orders)
end
"""
Takes MultiBernsteinImp and vector of scalar and returns new matrix
"""
function scalar_mul(
    multi::MultiBernsteinImp,
    scalars::TN;
) where {N<:Number,TN<:AbstractArray{N}}
    C = copy(multi.coefficient_matrix)

    rows, cols = size(C)


    j = 1
    t = 0
    for i = 1:length(multi.orders[1]):rows
        C[i, :] .*= scalars[j]
        t+=1
        if (t == multi.t[j])
            j+=1
            t=0
        end
    end

    return C
end
"""
element wise multiplication of the polynomials in a and b 
"""
function multiply(multi_a::MultiBernsteinImp, multi_b::MultiBernsteinImp;use_memory_optimizations=true)
    @assert length(multi_a.t) == length(multi_b.t)
    new_ts = multi_a.t .* multi_b.t
    n = length(multi_a.orders[1])
    max_res_order = maximum(maximum.(multi_a.orders .+ multi_b.orders))
    res = Matrix{eltype(multi_a.coefficient_matrix)}[]
    poly_a_offset = 0
    poly_b_offset = 0
    for poly_idx in eachindex(multi_a.t)
        C_a = binomial_tensor(multi_a.orders[poly_idx])
        C_b = binomial_tensor(multi_b.orders[poly_idx])
        max_order_poly_a = maximum(multi_a.orders[poly_idx])
        max_order_poly_b = maximum(multi_b.orders[poly_idx])
        C_rescale = binomial_tensor(multi_a.orders[poly_idx] .+ multi_b.orders[poly_idx])
	    res_poly = similar(multi_a.coefficient_matrix, new_ts[poly_idx]*n, max_res_order + 1)
	    offset = 1
        for term_a_idx = 0:(multi_a.t[poly_idx]-1)
            term_a = multi_a.coefficient_matrix[
                (poly_a_offset+1+(term_a_idx*n)):(poly_a_offset+(term_a_idx+1)*n),
                1 : max_order_poly_a +1
            ]
            for term_b_idx = 0:(multi_b.t[poly_idx]-1)
                term_b = multi_b.coefficient_matrix[
                    (poly_b_offset+1+(term_b_idx*n)):(poly_b_offset+(term_b_idx+1)*n),
                    1: max_order_poly_b + 1
                ]
                scaled_a = term_a .* C_a
                scaled_b = term_b .* C_b
                conv = row_convolution_kernel(scaled_a, scaled_b)[
                    :,
                    1:(maximum(multi_a.orders[poly_idx] .+ multi_b.orders[poly_idx])+1),
                ]
		        res_poly[offset: offset + size(conv,1)-1 , 1:size(conv,2) ] .= ifelse.(C_rescale .!= 0, conv ./ C_rescale, 0.0)
                if size(conv,2) != size(C_rescale,2)
		            res_poly[offset: offset + size(conv,1)-1, size(conv,2): end] .= 0.0
                end
		        offset+=size(conv,1)
            end
        end
        combined= res_poly
        if use_memory_optimizations 
            combined = combine_terms(res_poly,n)
        end
        cur_rows = size(res, 1)
        if isempty(res)
            res= combined 
        else
            res = vcat(res,combined) 
        end
        new_ts[poly_idx] = (size(res,1)- cur_rows) ÷ n
        poly_a_offset += n*multi_a.t[poly_idx]
        poly_b_offset += n*multi_b.t[poly_idx]
    end
    return MultiBernsteinImp(res, new_ts, multi_a.orders .+ multi_b.orders)
end
function square(multi::MultiBernsteinImp; use_memory_optimizations=true)
    new_ts = multi.t .* multi.t
    n = length(multi.orders[1])
    max_res_order = maximum(maximum.(multi.orders .+ multi.orders))
    res = Matrix{eltype(multi.coefficient_matrix)}[]
    poly_a_offset = 0
    poly_b_offset = 0
    for poly_idx in eachindex(multi.t)
        C = binomial_tensor(multi.orders[poly_idx])
        max_order_poly = maximum(multi.orders[poly_idx])
        C_rescale = binomial_tensor(multi.orders[poly_idx] .+ multi.orders[poly_idx])
	    res_poly = similar(multi.coefficient_matrix, new_ts[poly_idx]*n, max_res_order + 1)
	    offset = 1
        for term_a_idx = 0:(multi.t[poly_idx]-1)
            term_a = multi.coefficient_matrix[
                (poly_a_offset+1+(term_a_idx*n)):(poly_a_offset+(term_a_idx+1)*n),
                1 : max_order_poly +1
            ]
            scaled_a = term_a .* C
            for term_b_idx = 0:(multi.t[poly_idx]-1)
                term_b = multi.coefficient_matrix[
                    (poly_b_offset+1+(term_b_idx*n)):(poly_b_offset+(term_b_idx+1)*n),
                    1: max_order_poly + 1
                ]
                scaled_b = term_b .* C
                conv = row_convolution_kernel(scaled_a, scaled_b)[
                    :,
                    1:(maximum(multi.orders[poly_idx] .+ multi.orders[poly_idx])+1),
                ]
		res_poly[offset: offset + size(conv,1)-1 , 1:size(conv,2) ] .= ifelse.(C_rescale .!= 0, conv ./ C_rescale, 0.0)
                if size(conv,2) != size(C_rescale,2)
		            res_poly[offset: offset + size(conv,1)-1, size(conv,2): end] .= 0.0
                end
		offset+=size(conv,1)
            end
        end
        combined= res_poly
        if use_memory_optimizations 
            combined = combine_terms(res_poly,n)
        end
        cur_rows = size(res, 1)
        if isempty(res)
            res= combined 
        else
            res = vcat(res,combined) 
        end
        new_ts[poly_idx] = (size(res,1)- cur_rows) ÷ n
        poly_a_offset += n*multi.t[poly_idx]
        poly_b_offset += n*multi.t[poly_idx]
    end
    return MultiBernsteinImp(res, new_ts, multi.orders .+ multi.orders)
end

"""
Computes a one-dimensional quadratic map for each input dimension.
I.e. yᵢ =  qᵢxᵢ² for each input dimension i
"""
function quadratic_map_1d(qs, multi::MultiBernsteinImp; use_memory_optimizations=true)::MultiBernsteinImp
    quad = square(multi;use_memory_optimizations)
    return  MultiBernsteinImp(scalar_mul(quad,qs),quad.t,quad.orders)
end


"""
Computes a one-dimensional quadratic function for each input dimension.
I.e. yᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ for each input dimension i
"""
function quadratic_propagation(a, b, c, multi::MultiBernsteinImp; use_memory_optimizations=true)

    p_quad = quadratic_map_1d(a, multi; use_memory_optimizations)
    lin =  MultiBernsteinImp(scalar_mul(multi,b), multi.t,multi.orders)

    sum = add(lin, p_quad; use_memory_optimizations)
    
    return translate(sum,c)
    
end

function bounds(multi::MultiBernsteinImp; use_shortcut = true) 
    lbs = similar(multi.coefficient_matrix, size(multi.t))
    ubs = similar(multi.coefficient_matrix, size(multi.t))
    

    start = 1 
    n = length(multi.orders[1])
    println("starting bounds")
    start_time = time()
    @show nthreads()
    @threads for p_idx in eachindex(multi.t)
	if use_shortcut
	    lbs[p_idx], ubs[p_idx] = quadrant_ibf_minmax(multi.coefficient_matrix[start : start + (n * multi.t[p_idx])  - 1 , :  ],multi.t[p_idx],multi.orders[p_idx])
	else
	    lbs[p_idx], ubs[p_idx]= dense_min_max_threaded(multi.coefficient_matrix[start : start + (n * multi.t[p_idx])  - 1 , :  ],multi.t[p_idx],multi.orders[p_idx])
	end
        start += (n*multi.t[p_idx])
    end
    println("bounds took: $(time()- start_time)s")
    #@show lbs 
    #@show ubs 
    return lbs, ubs #, lbs_new_method, ubs_new_method, lbs_new_method_1, ubs_new_method_1
end

"""
Calculates concrete bounds for A*s + b for BernsteinPoly s with common generators.
"""
function bounds(A::AbstractMatrix, b::AbstractVector, s::BernsteinInterval; use_shortcut=true)
    mapped_interval = interval_map(
        min.(0, A),
        max.(0, A),
        s,
        b,
    )
    ll, lu = bounds(mapped_interval.Low; use_shortcut)
    ul, uu = bounds(mapped_interval.Up; use_shortcut)
    return ll, uu
end

function combine_terms(coefficient_matrix::TN, n::Int)where {N<:Number,TN<:AbstractArray{N}}

    nblocks = size(coefficient_matrix, 1) ÷ n

    # hash => representative block start rows
    groups = Dict{UInt, Vector{Int}}()

    # start row of each unique block
    representatives = Int[]

    for block in 0:nblocks-1

        row_start = block*n + 1

        tail = @view coefficient_matrix[row_start+1:row_start+n-1, :]

        h = hash(tail)

        found = false

        for rep_start in get(groups, h, Int[])

            rep_tail = @view coefficient_matrix[rep_start+1:rep_start+n-1, :]
            scalar_row =  coefficient_matrix[rep_start,: ] ./ coefficient_matrix[row_start,:]
            if tail == rep_tail &&  all(y->y==scalar_row[1],  scalar_row)
                @views coefficient_matrix[rep_start, :] .+= coefficient_matrix[row_start, :]
                found = true
                break
            end
        end

        if !found
            push!(get!(()->Int[], groups, h), row_start)
            push!(representatives, row_start)
        end
    end

    # Build output
    n_unique = length(representatives)
    result = TN(undef, n_unique*n, size(coefficient_matrix, 2))

    out_row = 1
    for rep_start in representatives

        rows = rep_start:rep_start+n-1

        @views result[out_row:out_row+n-1, :] .= coefficient_matrix[rows, :]

        out_row += n
    end
    println("removed $(nblocks- (size(result, 1) ÷ n))")
    return result
end

"""
per polynomial combine terms that consist of the same variables. 
"""
function combine_terms(multi::MultiBernsteinImp)
    n = length(multi.orders[1])
    columns = maximum(maximum.(multi.orders))+1
    out_chunks = Matrix{eltype(multi.coefficient_matrix)}[]
    
    new_terms = Int[]

    offset = 1
    for poly_idx in eachindex(multi.t)
	    coeff_mat = multi.coefficient_matrix[offset: offset+ multi.t[poly_idx]*n - 1,:]
	    offset += multi.t[poly_idx]*n 
	    combined = combine_terms(coeff_mat,n)
	    push!(out_chunks, combined)
	    push!(new_terms, size(combined,1) ÷ n)
    end

    return MultiBernsteinImp(vcat(out_chunks...),new_terms,multi.orders)
end
