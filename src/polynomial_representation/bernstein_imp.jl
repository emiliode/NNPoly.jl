
# multivariate bernstein polynomials 
using DynamicPolynomials, LazySets, SpecialFunctions, DSP, Base.Threads

struct BernsteinPolynomialImp{N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    coefficient_matrix::TN # matrix containing coefficients 
    n::M # input dimension
    t::M # number of terms
    X::Hyperrectangle{N}
    orders::Vector{Int64} # orders
end

function bernstein_basis(
    order::Int64,
    a::T,
    b::T,
)::Vector{AbstractPolynomialLike} where {T<:Number}
    basis = []
    @polyvar x
    for k = 0:order
        cur_basis = (binomial(order, k)/((b-a)^order)) * (x-a)^k * (b-x)^(order-k)
        push!(basis, cur_basis)
    end
    return basis
end

"""
Calculate bernstein coefficients for x^m
"""
function calculate_bern_coeff_for_monomial(e, n, a, b)
    #print("e=$e, a=$a, b=$b")
    # if e == 0 
    #    return zeros(n+1)
    #end
    alphas = zeros(n+1)
    for j = 0:n
        for m = j:n
            alphas[j+1] += m == e ? binomial(m, j) * a^(m-j) * (b-a)^j : 0
        end
    end
    coeffs = zeros(n+1)
    for k = 0:n
        for j = 0:k
            bin = (factorial(k) * factorial(n-j)) / (factorial(n) * factorial(k-j)) # more stable than bin(k,j)/bin(n,j)
            coeffs[k+1] += bin * alphas[j+1]
        end
    end
    return coeffs
end

"""
Calculate multivariate bernstein basis for a multivariate polynom with orders  degrees over input X  
"""
function get_basis_vectors(
    orders::Vector{Int64},
    X::Hyperrectangle{T},
)::Vector{Vector{AbstractPolynomialLike}} where {T<:Number}
    return [
        bernstein_basis(
            order,
            center(X)[i] - radius_hyperrectangle(X, i),
            center(X)[i]+radius_hyperrectangle(X, i),
        ) for (i, order) in enumerate(orders)
    ]
end
"""
Creates a BernsteinPolynomail from the given multivariate polynomial.
"""
function make_polynomial(
    polynom::AbstractPolynomialLike,
    orders::Vector{Int64},
    X::Hyperrectangle{T},
) where {T<:Number}
    #println("input: $polynom with orders: $orders over $X")
    basis_vector = get_basis_vectors(orders, X)
    #println("basisvector $basis_vector")
    all_bern_coeff = []
    for t in terms(polynom)
        ev = exponents(t)
        c = DynamicPolynomials.coefficient(t)
        #println("Term: $t | coeff: $c |EV: $ev ")
        for (i, e) in enumerate(ev)
            a = center(X)[i] - radius_hyperrectangle(X, i)
            b = center(X)[i] + radius_hyperrectangle(X, i)
            bern_coeff = calculate_bern_coeff_for_monomial(e, orders[i], a, b)
            if i == 1
                bern_coeff = c .* bern_coeff
            end
            bern_coeff = [bern_coeff; zeros(maximum(orders)-orders[i])]
            all_bern_coeff = push!(all_bern_coeff, bern_coeff)
        end
        #display(all_bern_coeff) 
    end
    bern_coeff_matrix = permutedims(stack(all_bern_coeff))
    #println("coeff Matrix:")
    #display(bern_coeff_matrix)
    return BernsteinPolynomialImp(
        bern_coeff_matrix,
        length(orders),
        length(terms(polynom)),
        X,
        orders,
    )
end

function evaluate(
    bern_imp::BernsteinPolynomialImp,
    point::AR,
) where {N<:Number,AR<:AbstractArray{N}}
    i = 0
    bern_eval = []
    for row in bern_imp.bernstein_basis
        push!(
            bern_eval,
            [
                [p(variables(p)[1] => point[i+1]) for p in row];
                zeros(maximum(bern_imp.orders)-bern_imp.orders[i+1])
            ],
        )
        i = (i + 1) % bern_imp.n
    end
    print("bern_eval: $bern_eval");

    eval = 0
    for term_idx = 0:(bern_imp.t-1)
        term_eval = 1
        for var_idx = 1:bern_imp.n
            row = @view bern_imp.coefficient_matrix[(term_idx*bern_imp.n+(var_idx)), :]

            bla = row .* bern_eval[var_idx]
            zws = sum(bla)
            println("Term: $term_idx, row: $row, var: $var_idx, $bla , $zws")
            term_eval *= zws
        end
        eval += term_eval
    end
    return eval
end

"""
Multiply a bernstein polynomial by a scalar
"""
function scalar_mul(bern_poly::BernsteinPolynomialImp, scalar::Number)
    # iterate over terms and multiply scalar into first monomial of the term
    new_coeff_mat = copy(bern_poly.coefficient_matrix)
    for term_idx = 0:(bern_poly.t-1)
        new_coeff_mat[(term_idx*bern_poly.n)+1, :] =
            bern_poly.coefficient_matrix[(term_idx*bern_poly.n)+1, :] .* scalar
    end
    return BernsteinPolynomialImp(
        new_coeff_mat,
        bern_poly.n,
        bern_poly.t,
        bern_poly.X,
        bern_poly.orders,
    )
end

"""
add constant to the polynomial
"""
function translate(poly::BernsteinPolynomialImp, b::Number)

    b_coeff_mat = fill(0.0, length(poly.orders), maximum(poly.orders)+1)
    for (i, order) in enumerate(poly.orders)
        b_coeff_mat[i, 1:(order+1)] .= 1
    end
    b_coeff_mat[1, :] .*= b
    return BernsteinPolynomialImp(
        vcat(poly.coefficient_matrix, b_coeff_mat),
        poly.n,
        poly.t+1,
        poly.X,
        poly.orders,
    )
end


"""
calculate : 
    [(l1 choose 0) , ... , (l1 choose l1)]
    ... 
    [(ln choose 0) , ... , (ln choose ln)] padded with zeros
"""
function binomial_tensor(L::Vector{Int64})
    ranges = [collect(0:i) for i in L]

    N = reshape(L, :, 1)

    # maximum sequence length
    maxlen = maximum(length.(ranges))

    # pad with -1
    R = fill(-1, length(ranges), maxlen)

    for (i, r) in enumerate(ranges)
        R[i, 1:length(r)] = r
    end

    A = loggamma.(N .+ 1)
    B = loggamma.(N .- R .+ 1)
    C = loggamma.(R .+ 1)
    return exp.(A .- B .- C)
end


function elevate_degree(
    bern_imp::BernsteinPolynomialImp,
    new_orders::Vector{Int64},
)::BernsteinPolynomialImp
    l_diff = new_orders - bern_imp.orders
    I_non_zero = findall(!iszero, l_diff)
    orders_non_zero = bern_imp.orders#[I_non_zero]
    l_diff_non_zero = l_diff#[I_non_zero]
    new_orders_non_zero = new_orders#[I_non_zero]


    pAmod = bern_imp.coefficient_matrix
    C_degree = binomial_tensor(orders_non_zero)
    C_degree = repeat(C_degree, bern_imp.t, 1)

    C_diff = binomial_tensor(l_diff_non_zero)
    C_diff = repeat(C_diff, bern_imp.t, 1)

    C_new_orders = binomial_tensor(new_orders_non_zero)
    #println("C_new_orders: $C_new_orders")
    C_new_orders = repeat(C_new_orders, bern_imp.t, 1)

    res = pAmod .* C_degree
    #println("pAmod:  $pAmod")
    #println("C_degree:  $C_degree")
    #println("res of mul:  $res")
    #println("C_diff :  $C_diff")
    result = row_convolution_kernel(res, C_diff)
    result = result[:, 1:(maximum(new_orders)+1)]
    #println("result of conv: $result")
    #println("C_new_orders: $C_new_orders")
    result = ifelse.(C_new_orders .!= 0, result ./ C_new_orders, 0.0)

    return BernsteinPolynomialImp(result, bern_imp.n, bern_imp.t, bern_imp.X, new_orders)

end
function row_convolution_kernel(T1::AbstractMatrix, T2::AbstractMatrix)
    T1_rows, T1_cols = size(T1)
    T2_rows, T2_cols = size(T2)

    @assert T1_rows == T2_rows "Row-wise convolution requires same number of rows"

    result_cols = T1_cols + T2_cols - 1
    result = zeros(eltype(T1), T1_rows, result_cols)

    for row = 1:T1_rows
        for col = 1:result_cols

            sum = zero(eltype(T1))

            start_k = max(1, col - T2_cols + 1)
            end_k = min(col, T1_cols)

            for k = start_k:end_k
                sum += T1[row, k] * T2[row, col-k+1]
            end

            result[row, col] = sum
        end
    end

    return result
end


function row_wise_conv(A, B)
    m, n1 = size(A)
    _, n2 = size(B)

    convolution = zeros(eltype(A), m, n1+n2 - 1)
    for i = 1:m
        convolution[1, :] = DSP.conv(A[i, :], B[i, :])
    end
    return convolution
end



"""
multiply two terms
"""
function multiply_term(
    term_a::Matrix,
    orders_a::Vector{Int64},
    term_b::Matrix,
    orders_b::Vector{Int64},
)::Matrix
    scaled_term_a = binomial_tensor(orders_a) .* term_a
    #println("scaled-term_a: $(scaled_term_a)")
    scaled_term_b = binomial_tensor(orders_b) .* term_b
    #println("scaled-term_b: $(scaled_term_b)")
    convolution = row_convolution_kernel(scaled_term_a, scaled_term_b)
    # keep only the relevant columns
    convolution = convolution[:, 1:(maximum(orders_a .+ orders_b)+1)]

    #println("conv: $(convolution)")
    rescaling_matrix = binomial_tensor(orders_a .+ orders_b)
    #println("rescaling_matrix: $(rescaling_matrix)")
    return ifelse.(rescaling_matrix .!= 0, convolution ./ rescaling_matrix, 0.0)
end

"""
Multiply bern_a and bern_b 
"""
function multiply(
    bern_a::BernsteinPolynomialImp,
    bern_b::BernsteinPolynomialImp,
)::BernsteinPolynomialImp
    res = []
    for i = 0:(bern_a.t-1)
        term_a = bern_a.coefficient_matrix[(i*bern_a.n+1):((i*bern_a.n)+bern_a.n), :]
        for j = 0:(bern_b.t-1)
            term_b = bern_b.coefficient_matrix[(j*bern_b.n+1):((j*bern_b.n)+bern_b.n), :]
            multiplied = multiply_term(term_a, bern_a.orders, term_b, bern_b.orders)
            #println("a[$i]*b[$j]=$multiplied")
            push!(res, multiplied)
        end
    end
    #println(res)
    return BernsteinPolynomialImp(
        reduce(vcat, res),
        bern_a.n,
        bern_a.t*bern_b.t,
        bern_a.X,
        bern_a.orders+bern_b.orders,
    )
end


"""
add two Bernstein polynomials in their Implicit representation
"""
function add(
    bern_a::BernsteinPolynomialImp,
    bern_b::BernsteinPolynomialImp,
)::BernsteinPolynomialImp
    res_orders = max.(bern_a.orders, bern_b.orders)
    elevated_bern_a = elevate_degree(bern_a, res_orders)
    elevated_bern_b = elevate_degree(bern_b, res_orders)

    return BernsteinPolynomialImp(
        vcat(elevated_bern_a.coefficient_matrix, elevated_bern_b.coefficient_matrix),
        elevated_bern_a.n,
        elevated_bern_a.t+elevated_bern_b.t,
        elevated_bern_a.X,
        res_orders,
    )
end


"""
Calculate bounds by evalueting the terms at (0,0,...) and (1,1,...)
"""
function quadrant_ibf_minmax(coefficient_matrix::TN, t::M, orders::Vector{Int64}, monomon_coeffs, inverse) where { M<:Number,O<:Number,TN<:AbstractArray{O}}

    

    @show monomon_coeffs
    nvars = length(orders)
    minval = zero(O)
    maxval = zero(O)

    for term in 0:(t-1)


        left_prod  = one(O)
        right_prod = one(O)
	    # first var has  to first be splitted because we combine ax* bx^2  + cx * bx^2 too (a+c)x + bx^2
	     

	for var in 2:nvars
	    coeffs = @view coefficient_matrix[term*nvars  + var, :]

	    first_idx = 1  #findfirst(!iszero, coeffs)
	    last_idx  = orders[var] +1  #findlast(!iszero, coeffs)

	    left_prod *= coeffs[first_idx]
	    right_prod *= coeffs[last_idx]
	end
	first_row = @view coefficient_matrix[term*nvars + 1,1:orders[1]+1] 

	if any([are_multiples(first_row , monomon_coefficient) for monomon_coefficient in monomon_coeffs])
	    left_prod *= first_row[1]
	    right_prod *= first_row[orders[1]+1]
	    @show left_prod, right_prod
	    minval += min(left_prod, right_prod)
	    maxval += max(left_prod, right_prod)
	    @show minval, maxval
	else 
	    #@show first_row
	    throw("SHOULD NOT HAPPEN")
	    original_coeffs = inverse * first_row
	    if all(original_coeffs .== 0 ) 
		continue 
	    end
	    for row in  monomon_coeffs .* original_coeffs
		first_idx = 1  
	    	last_idx  = orders[1] +1  

	    	left_prod_with_first_row = left_prod * row[first_idx]
	    	right_prod_with_first_row = right_prod * row[last_idx]
		minval += min(left_prod_with_first_row, right_prod_with_first_row)
		maxval += max(left_prod_with_first_row, right_prod_with_first_row)
	    end

	end

    end
    return minval, maxval
end

"""
Compute dense(idx) of a bernstein polynomial 
"""
function dense(bern_poly::BernsteinPolynomialImp, idx::CartesianIndex)::Number
    return dense(bern_poly.coefficient_matrix, bern_poly.t, bern_poly.orders, idx)
end
function dense(coefficient_matrix::TN,t::M, orders::Vector{Int64}, idx::CartesianIndex)::Number where { M<:Number,O<:Number,TN<:AbstractArray{O}}
    n = length(orders)

    res = 0
    for term_idx = 0:(t-1)
        prod = 1
        for dim = 1:n
            #print("($(term_idx*bern_poly.n +dim), $(dim))=$(bern_poly.coefficient_matrix[(term_idx*bern_poly.n+dim),idx[dim]])")
            prod *= coefficient_matrix[(term_idx*n+dim), idx[dim]]
        end
        res += prod
    end
    return res
end

function dense_min_max(coefficient_matrix::TN, t::M, orders::Vector{Int64}) where { M<:Number,O<:Number,TN<:AbstractArray{O}} 
    min = Inf
    max = -Inf
    for I in CartesianIndices(Tuple(orders .+ 1))
        cur_val = dense(coefficient_matrix,t,orders, I)
        if cur_val < min 
            min = cur_val
        end
        if cur_val > max 
            max = cur_val
        end
    end
    return min, max
end
function dense_min_max_threaded(
    coefficient_matrix::TN,
    t::M,
    orders::Vector{Int64}
) where {M<:Number,O<:Number,TN<:AbstractArray{O}}

    CI = CartesianIndices(Tuple(orders .+ 1))

    local_min = fill(Inf, nthreads())
    local_max = fill(-Inf, nthreads())

    @threads for idx in eachindex(CI)

        tid = threadid()

        cur_val = dense(
            coefficient_matrix,
            t,
            orders,
            CI[idx]
        )

        local_min[tid] = min(local_min[tid], cur_val)
        local_max[tid] = max(local_max[tid], cur_val)
    end

    return minimum(local_min), maximum(local_max)
end

function dense(coefficient_matrix::TN,t::M, orders::Vector{Int64}) where { M<:Number,O<:Number,TN<:AbstractArray{O}} 
    dense_tensor = zeros(Tuple(orders .+ 1))
    for I in CartesianIndices(dense_tensor)
        dense_tensor[I] = dense(coefficient_matrix,t,orders, I)
    end
    return dense_tensor
end
function dense(bern_poly::BernsteinPolynomialImp)
    dense_tensor = zeros(Tuple(bern_poly.orders .+ 1))
    for I in CartesianIndices(dense_tensor)
        dense_tensor[I] = dense(bern_poly, I)
    end
    return dense_tensor
end
