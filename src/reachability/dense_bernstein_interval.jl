using FFTW, DSP
struct DenseBernsteinInterval{N<:Number,O<:Number,TN<:AbstractArray{O},VN<:AbstractArray{N}}
    Low::TN # tensor with dimesions: l1 x ... x l_n x num_polys 
    Up::TN
    X::Hyperrectangle{N}
    lbs::Vector{VN}
    ubs::Vector{VN}
end

function init_one_coeff_mat(pos,X::Hyperrectangle, i::Int,n_dims::Int)

    shape = ntuple(i -> i ==  pos ? 2 : 1, n_dims)
    return reshape([low(X)[i], high(X)[i]], shape...) .* ones(eltype(X.radius), ntuple(_ -> 2, n_dims)...)
end


function init_dense_bernstein_interval(h::Hyperrectangle)
    num_polys = dim(h)
    unfixed_mask = (h.radius .!= 0)
    n = count(unfixed_mask)
    #@show unfixed_mask,n
    @polyvar x[1:num_polys]
    coeff_matrices = ones(eltype(h.radius), [2 for _ in 1:n]... ,num_polys  )
    
    idx = ntuple(_ -> Colon(), n)   # (:, :, ..., :)  -- n mal
    for i in 1:num_polys
	    if unfixed_mask[i]
	        coeff_matrices[idx... ,i]  = init_one_coeff_mat(count(@view unfixed_mask[1:i]),h,i,n)
	    else 
	        # constant value
	        coeff_mat = fill(low(h)[i],ntuple(_ ->2, n))
	        coeff_matrices[idx...,i] = coeff_mat
	    end
    end
    X = Hyperrectangle(h.center[unfixed_mask], h.radius[unfixed_mask])
    return DenseBernsteinInterval(
        coeff_matrices,
        copy(coeff_matrices),
        X,
        Vector{eltype(h.radius)}[],
        Vector{eltype(h.radius)}[]
    )
end
function batched_square_conv_rowwise(A::AbstractArray)
    # A: (rows, cols)
    
    n_rows, n_cols = size(A)
    
    out_cols = n_cols + kernel_size - 1
    T = promote_type(eltype(A), eltype(B))
    
    padA = zeros(T, n_rows, out_cols)
    padB = zeros(T, n_rows, out_cols)
    
    padA[:, 1:n_cols] .= A
    padB[:, 1:kernel_size] .= B
    
    # FFT über Spalten (Dimension 2) für jede Reihe
    FA = fft(padA, [2])
    FB = fft(padB, [2])
    
    # Element-weise Multiplikation
    Y_fft = FA .* FB
    
    # IFFT
    Y = real.(ifft(Y_fft, [2]))
    
    return Y
end
function batched_conv_rowwise(A::AbstractArray, B::AbstractArray)
    # A: (rows, cols)
    # B: (rows, kernel_size)
    
    n_rows, n_cols = size(A)
    _, kernel_size = size(B)
    
    @assert size(B, 1) == n_rows "B must have same number of rows as A"
    
    out_cols = n_cols + kernel_size - 1
    T = promote_type(eltype(A), eltype(B))
    
    padA = zeros(T, n_rows, out_cols)
    padB = zeros(T, n_rows, out_cols)
    
    padA[:, 1:n_cols] .= A
    padB[:, 1:kernel_size] .= B
    
    # FFT über Spalten (Dimension 2) für jede Reihe
    FA = fft(padA, [2])
    FB = fft(padB, [2])
    
    # Element-weise Multiplikation
    Y_fft = FA .* FB
    
    # IFFT
    Y = real.(ifft(Y_fft, [2]))
    
    return Y
end
function batched_conv(A::AbstractArray, B::AbstractArray)
    poly_dims_a = size(A)[1:end-1]
    poly_dims_b = size(B)[1:end-1]
    @show size(A), size(B)
    n_a = size(A)[end]
    n_b = size(B)[end]
    @assert n_b == 1 || n_a == n_b "batch dim of B must be 1 (broadcast) or match A"

    out_dims = poly_dims_a .+ poly_dims_b .- 1
    fft_dims = ntuple(identity, length(out_dims))
    T = promote_type(eltype(A), eltype(B))

    padA = zeros(T, out_dims..., n_a)
    padB = zeros(T, out_dims..., n_b)
    padA[axes(A)[1:end-1]..., :] .= A
    padB[axes(B)[1:end-1]..., :] .= B

    FA = fft(padA, fft_dims)
    FB = fft(padB, fft_dims)
    return real.(ifft(FA .* FB, fft_dims))
end
function ChainRulesCore.rrule(::typeof(batched_conv), A::AbstractArray, B::AbstractArray)
    # --- Forward Pass (Exact same logic as the original function) ---
    poly_dims_a = size(A)[1:end-1]
    poly_dims_b = size(B)[1:end-1]
    @show size(A), size(B)
    n_a = size(A)[end]
    n_b = size(B)[end]
    @assert n_b == 1 || n_a == n_b "batch dim of B must be 1 (broadcast) or match A"

    out_dims = poly_dims_a .+ poly_dims_b .- 1
    fft_dims = ntuple(identity, length(out_dims))
    T = promote_type(eltype(A), eltype(B))

    padA = zeros(T, out_dims..., n_a)
    padB = zeros(T, out_dims..., n_b)
    padA[axes(A)[1:end-1]..., :] .= A
    padB[axes(B)[1:end-1]..., :] .= B

    FA = fft(padA, fft_dims)
    FB = fft(padB, fft_dims)
    
    Y = real.(ifft(FA .* FB, fft_dims))

    project_A = ProjectTo(A)
    project_B = ProjectTo(B)

    # --- Backward Pass (Pullback closure) ---
    function batched_conv_pullback(ΔY_raw)
        # Unthunk the incoming gradient (Zygote might pass a Thunk)
        ΔY = unthunk(ΔY_raw)
        
        # FFT of the incoming output gradient
        FΔY = fft(ΔY, fft_dims)

        # 1. Gradient for A: Cross-correlation of ΔY with B
        # ∇A_padded = ifft( fft(ΔY) .* conj(fft(B)) )
        ∇A_padded = real.(ifft(FΔY .* conj.(FB), fft_dims))
        # Crop to the original spatial shape of A
        ∇A = ∇A_padded[axes(A)...]

        # 2. Gradient for B: Cross-correlation of ΔY with A
        ∇B_padded = real.(ifft(FΔY .* conj.(FA), fft_dims))
        ∇B_full = ∇B_padded[axes(B)[1:end-1]..., :]
        
        # If B was broadcasted (n_b == 1), we must sum the gradients over the batch dimension
        if n_b == 1 && n_a > 1
            ∇B = sum(∇B_full, dims=ndims(B))
        else
            ∇B = ∇B_full
        end

        # Return NoTangent() for the function itself, then gradients for A and B
        return NoTangent(), project_A(∇A), project_B(∇B)
    end

    return Y, batched_conv_pullback
end

function batched_square_conv(A::AbstractArray)
    poly_dims = size(A)[1:end-1]
    out_dims = 2 .* poly_dims .- 1
    fft_dims = ntuple(identity, length(out_dims))
    padA = zeros(eltype(A), out_dims..., size(A)[end])
    padA[axes(A)[1:end-1]..., :] .= A
    FA = fft(padA, fft_dims)
    return real.(ifft(FA.^2, fft_dims))   # nur EINE FFT statt zwei
end
function ChainRulesCore.rrule(::typeof(batched_square_conv), A::AbstractArray)
    # --- Forward Pass ---
    poly_dims = size(A)[1:end-1]
    out_dims = 2 .* poly_dims .- 1
    fft_dims = ntuple(identity, length(out_dims))
    
    padA = zeros(eltype(A), out_dims..., size(A)[end])
    padA[axes(A)[1:end-1]..., :] .= A
    
    FA = fft(padA, fft_dims)
    Y = real.(ifft(FA.^2, fft_dims))
    
    project_A = ProjectTo(A)

    # --- Backward Pass ---
    function batched_square_conv_pullback(ΔY_raw)
        ΔY = unthunk(ΔY_raw)
        
        # FFT of the incoming gradient
        FΔY = fft(ΔY, fft_dims)
        
        # ∇A = 2 * (ΔY ★ A)
        ∇A_padded = 2 .* real.(ifft(FΔY .* conj.(FA), fft_dims))
        
        # Crop back to the original size of A
        ∇A = ∇A_padded[axes(A)...]
        
        return NoTangent(), project_A(∇A)
    end
    
    return Y, batched_square_conv_pullback
end

function multiply(dense_a::AbstractArray, dense_b::AbstractArray) 
    @assert size(dense_a)[end] == size(dense_b)[end] "both matrices must contain the same number of polynomials"
    orders_a = [size(dense_a)[1:end - 1]...] .- 1
    orders_b = [size(dense_b)[1:end - 1]...] .- 1
    num_polys = size(dense_a)[end]
    res_orders = orders_a .+ orders_b
    C_a = binomial_tensor_dense_cached(orders_a)
    C_b = binomial_tensor_dense_cached(orders_b)

    scaled_a = dense_a .* C_a
    scaled_b = dense_b .* C_b

    #res_conv = stack([DSP.conv(scaled_a[ ntuple(_ -> Colon(),ndims(scaled_a)-1)..., i ] , scaled_b[ ntuple(_ -> Colon(),ndims(scaled_b)-1)..., i ] ) for i in 1:num_polys ])
    res_conv = batched_conv(scaled_a,scaled_b)
    #@assert isapprox(res_conv,res_conv_batch)
    C_rescale = binomial_tensor_dense_cached(res_orders)

	return  res_conv ./ C_rescale
end

function square(dense::AbstractArray)
    println("HELLO")
    orders = get_orders(dense)
    num_polys = size(dense)[end]
    res_orders = 2 .* orders 
    C_a = binomial_tensor_dense_cached(orders)
    C_rescale = binomial_tensor_dense_cached(res_orders)
    scaled = dense .* C_a
    #res_conv = stack([DSP.conv(scaled[ ntuple(_ -> Colon(),ndims(scaled)-1)..., i ] , scaled[ ntuple(_ -> Colon(),ndims(scaled)-1)..., i ] ) for i in 1:num_polys ])
    res_conv= batched_square_conv(scaled)

    #@assert isapprox(res_conv,res_conv_batch)

	return  res_conv ./ C_rescale
end

"""
returns the orders of the stored polynomials in the dense tensor.
"""
function get_orders(dense::AbstractArray)::Vector{Int64}
    return [size(dense)[1:end - 1]...] .- 1
end
function elevate(dense::AbstractArray,new_orders::Vector{Int64})
    cur_orders = get_orders(dense)
    diffs = new_orders .- cur_orders  
    C_cur = binomial_tensor_dense_cached(cur_orders)
    C_diff = binomial_tensor_dense_cached(diffs)
    C_new_orders = binomial_tensor_dense_cached(new_orders)

    scaled_dense = dense .* C_cur
    #res_conv = stack([DSP.conv(scaled_dense[ ntuple(_ -> Colon(),ndims(scaled_dense)-1)..., i ] , C_diff ) for i in axes(scaled_dense, ndims(scaled_dense)) ])
    res_conv= batched_conv(scaled_dense,  reshape(C_diff, size(C_diff)...,1))
    #@assert isapprox(res_conv,res_conv_batch)

    return res_conv ./ C_new_orders
end

function add(dense_a::AbstractArray, dense_b::AbstractArray)
    orders_a = get_orders(dense_a)
    orders_b = get_orders(dense_b)
    new_orders = max.(orders_a, orders_b)

    elevated_a = orders_a != new_orders ?  elevate(dense_a,new_orders) : dense_a
    elevated_b = orders_b != new_orders ?  elevate(dense_b,new_orders) : dense_b

    return elevated_a .+ elevated_b

end

function add(inter_a::DenseBernsteinInterval, inter_b::DenseBernsteinInterval)
    return DenseBernsteinInterval(add(inter_a.Low, inter_b.Low), add(inter_a.Up, inter_b.Up),inter_a.X,inter_a.lbs,inter_b.ubs)
end

function scalar_mul(dense_tensor::AbstractArray, vec::AbstractVector)
    shape = [ntuple(_ -> 1, ndims(dense_tensor) - 1)..., size(dense_tensor)[end]]

    return dense_tensor .* reshape(vec, shape...)
end


function translate(inter::DenseBernsteinInterval,b::AbstractVector) 
    return translate(inter,b,b)
end
function translate(inter::DenseBernsteinInterval,b_low::AbstractVector,b_up::AbstractVector) 
    shape = [ntuple(_ -> 1, ndims(inter.Low) - 1)..., size(inter.Low)[end]]
    return DenseBernsteinInterval(inter.Low .+ reshape(b_low,shape...), inter.Up .+ reshape(b_up,shape...),inter.X,inter.lbs,inter.ubs)
end

"""
Computes a one-dimensional quadratic map for each input dimension.
I.e. yᵢ =  qᵢxᵢ² for each input dimension i
"""
function quadratic_map_1d(qs_low, qs_up, interval::DenseBernsteinInterval; use_memory_optimizations=true)::DenseBernsteinInterval
    quad_low = square(interval.Low)
    quad_up = square(interval.Up)

    lin_low = scalar_mul(quad_low, qs_low)
    lin_up = scalar_mul(quad_up, qs_up)
    return DenseBernsteinInterval(lin_low,lin_up,interval.X,interval.lbs,interval.ubs )
end


"""
Computes a one-dimensional quadratic function for each input dimension.
I.e. yᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ for each input dimension i
"""
function quadratic_propagation(a_low,  b_low, c_low,a_up, b_up, c_up, inter::DenseBernsteinInterval; use_memory_optimizations=true)

    p_quad = quadratic_map_1d(a_low,a_up, inter)
    lin_low_coeffs = scalar_mul(inter.Low , b_low)
    lin_up_coeffs = scalar_mul(inter.Up , b_up)

    sum =  add( DenseBernsteinInterval(lin_low_coeffs,lin_up_coeffs,inter.X,inter.lbs,inter.ubs), p_quad)
    
    return translate(sum,c_low,c_up)
    
end

function linear_map(W::AbstractArray, dense::AbstractArray)
    poly_shape = size(dense)[1:end - 1]
    n = size(dense)[end]
    m = size(W,1)
    T_flat = reshape(dense, prod(poly_shape),n) 
    result_flat = T_flat * W'  # W' because reshape uses julias column major layout
    return reshape(result_flat, poly_shape...,m)
end
"""
Interval map overapproximating an affine map Wx + b for x ∈ I 

W⁻ - (matrix) negative weights
W⁺ - (matrix) positive weights
I  - (BernsteinInterval) bernsteinInterval
b  - (vector) bias
"""
function interval_map(W⁻, W⁺, I::DenseBernsteinInterval, b; use_memory_optimizations=true)
    #@show W⁻ , W⁺
    new_low = linear_map(W⁻ , I.Up) .+ linear_map(W⁺ , I.Low)
    new_up = linear_map(W⁻ , I.Low) .+ linear_map(W⁺ , I.Up)
    return translate(DenseBernsteinInterval(new_low,new_up, I.X, I.lbs,I.ubs),b)
end

function bounds(inter::DenseBernsteinInterval; use_shortcut)
    dims = ntuple(identity,ndims(inter.Low)-1)
    last_dim = size(inter.Low)[end]

    llbs = reshape(minimum(inter.Low; dims),last_dim)
    lubs = reshape(maximum(inter.Low; dims),last_dim)
    ulbs = reshape(minimum(inter.Up; dims),last_dim)
    uubs = reshape(maximum(inter.Up; dims),last_dim)
    return llbs, lubs, ulbs, uubs
end
function outer_bounds(inter::DenseBernsteinInterval; use_shortcut)
    dims = ntuple(identity,ndims(inter.Low)-1)
    last_dim = size(inter.Low)[end]
    llbs = reshape(minimum(inter.Low; dims),last_dim)
    uubs = reshape(maximum(inter.Up; dims),last_dim)
    return llbs,  uubs
end
function bounds(tensor::AbstractArray; use_shortcut) 
    dims = ntuple(identity,ndims(tensor)-1)
    last_dim = size(tensor)[end]
    ext = extrema(tensor;dims)
    lbs = reshape(first.(ext),last_dim)
    ubs = reshape(last.(ext),last_dim)
    return lbs, ubs

end

"""
Calculates concrete bounds for A*s + b for BernsteinPoly s with common generators.
"""
function bounds(A::AbstractMatrix, b::AbstractVector, s::DenseBernsteinInterval; use_shortcut=true)
    mapped_interval = interval_map(
        min.(0, A),
        max.(0, A),
        s,
        b,
    )
    include("./../debug/helper.jl")
    print_dense_interval(mapped_interval)
    ll, lu ,ul, uu = bounds(mapped_interval; use_shortcut)
    @show ll,lu,ul, uu
    return outer_bounds(mapped_interval; use_shortcut)
end
