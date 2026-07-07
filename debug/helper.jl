using NNPoly, LazySets, DynamicPolynomials, Test, NeuralVerification
NP = NNPoly 
NV = NeuralVerification

function forward(net::NV.Chain, x)
    a = x
    for layer in net.layers
        a = layer.activation.(layer.weights * a .+ layer.bias)
    end
    return a
end
function point_in(input_set::AbstractHyperrectangle)
    lb = low(input_set)
    ub = high(input_set)
    return lb .+ Base.rand(length(lb)) .* (ub .- lb)
end
function dense_to_monomon(bern_dense, bern_implicit::NP.BernsteinPolynomialImp)
    @polyvar x y
    res = 0
    basis_vectors = []
    push!(
        basis_vectors,
        [subs(p, variables(p)[1] => x) for p in bern_implicit.bernstein_basis[1]],
    )
    push!(
        basis_vectors,
        [subs(p, variables(p)[1] => y) for p in bern_implicit.bernstein_basis[2]],
    )
    for I in CartesianIndices(bern_dense)
        print("I=$I")
        prod = bern_dense[I]
        for i = 1:bern_implicit.n

            println("\ti=$i")
            basis = basis_vectors[i][I[i]]
            println("\tbasis=$basis")
            prod*=basis
        end
        res += prod
    end
    return res
end

function prune_sparse(p; tol = 1e-5)
    #println(p)
    mons = monomials(p)
    coeffs = coefficients(p)

    keep = abs.(coeffs) .≥ tol
    #println(keep)
    if !any(keep)
        return 0
    end
    return sum(c*m for (c, m) in zip(mons[keep], coeffs[keep]))
end

function imp_to_monomon(bern_imp::NP.BernsteinPolynomialImp, x)
    bern_eval = []
    bernstein_basis = NP.get_basis_vectors(bern_imp.orders, bern_imp.X)
    for i in eachindex(x)
        push!(
            bern_eval,
            [
                [p(variables(p)[1] => x[i]) for p in bernstein_basis[i]];
                zeros(maximum(bern_imp.orders)-bern_imp.orders[i])
            ],
        )
    end
    #  print("bern_eval: $bern_eval");

    eval = 0
    for term_idx = 0:(bern_imp.t-1)
        term_eval = 1
        for var_idx = 1:bern_imp.n
            row = @view bern_imp.coefficient_matrix[(term_idx*bern_imp.n+(var_idx)), :]

            bla = row .* bern_eval[var_idx]
            #println(bla)
            zws = sum(bla, init = 0.0)
            #println("Term: $term_idx, row: $row, var: $var_idx, $bla , $zws")
            # must result in monomial term therefore round to zero
            term_eval *= prune_sparse(zws)
        end
        eval += term_eval
    end
    return eval
end
function imp_to_monomon(
    coeff_mat::TN,
    n::Int64,
    t::Int64,
    orders::Vector{Int64},
    X::Hyperrectangle,
    x,
) where {N<:Number,TN<:AbstractArray{N}}
    bern_eval = []
    bernstein_basis = NP.get_basis_vectors(orders, X)
    #@show bernstein_basis,orders
    for i = 1:length(orders)
        push!(
            bern_eval,
            [
                [p(variables(p)[1] => x[i]) for p in bernstein_basis[i]];
                zeros(maximum(orders)-orders[i])
            ],
        )
    end
    #@show bern_eval
    #  print("bern_eval: $bern_eval");

    eval = 0
    for term_idx = 0:(t-1)
        term_eval = 1
        for var_idx = 1:n
            row = @view coeff_mat[(term_idx*n+(var_idx)), :]

            bla = row .* bern_eval[var_idx]
            #println(bla)
            zws = sum(bla, init = 0.0)
            #println("Term: $term_idx, row: $row, var: $var_idx, $bla , $zws")
            # must result in monomial term therefore round to zero
            term_eval *= prune_sparse(zws)
        end
        eval += term_eval
    end
    return eval
end
function print_multi_bernstein_imp(multi::NP.MultiBernsteinImp, n, X)
    @polyvar x[1:n]
    start = 1
    for i = 1:(length(multi.t))
        println(
            "$i:",
            imp_to_monomon(
                multi.coefficient_matrix[start:(start+multi.t[i]*n-1), 1:maximum(multi.orders)+1],
                n,
                multi.t[i],
                multi.orders,
                X,
                x,
            ),
        )
        start += multi.t[i] * n
    end
end
function print_combined_inter(inter::NP.CombinedPolyBernsteinInterval;show_matrices=false)
    @polyvar x[1:inter.n]
    println("bern_terms")
    show_matrices &&@show inter.bern_terms
    for i in 1:inter.t
	term_start = (i-1)*inter.n + 1
	println("$i:", imp_to_monomon(
			    inter.bern_terms[ term_start : i* inter.n, :],
			    inter.n,
			    1,
			    inter.orders,
			    inter.X,
			    x))
    end
    println("polys Low")
    show_matrices && @show inter.Low
    for i in  axes(inter.Low,1)
	println("$i:", imp_to_monomon(
			    NP.get_poly_low(i,inter),
			    inter.n,
			    inter.t,
			    inter.orders,
			    inter.X,
			    x))
    end
    println("polys Up")
    show_matrices && @show inter.Up
    for i in  axes(inter.Low,1)
	println("$i:", imp_to_monomon(
			    NP.get_poly_up(i,inter),
			    inter.n,
			    inter.t,
			    inter.orders,
			    inter.X,
			    x))
	
	
    end
end
function print_combined_interval(bern_interval::NP.CombinedPolyBernsteinInterval;show_matrices=false) 
    @polyvar x[1:bern_interval.Low.n]
    println("bern_terms")
    show_matrices &&@show bern_interval.Low.bern_terms
    show_matrices &&@show bern_interval.Up.bern_terms
    for i in 1:bern_interval.Low.t
	term_start = (i-1)*bern_interval.Low.n + 1
	println("$i:", imp_to_monomon(
			    bern_interval.Low.bern_terms[term_start : i* bern_interval.Low.n, :],
			    bern_interval.Low.n,
			    1,
			    bern_interval.Low.orders,
			    bern_interval.Low.X,
			    x))
    end
    for i in 1:bern_interval.Up.t
	term_start = (i-1)*bern_interval.Up.n + 1
	println("$i:", imp_to_monomon(
			    bern_interval.Up.bern_terms[term_start : i* bern_interval.Up.n, :],
			    bern_interval.Up.n,
			    1,
			    bern_interval.Up.orders,
			    bern_interval.Up.X,
			    x))
    end
    println("Lower polys")
    show_matrices && @show bern_interval.Low.coeffs
    for i in  axes(bern_interval.Low.coeffs,1)
	println("$i:", imp_to_monomon(
			    NP.get_poly(i,bern_interval.Low),
			    bern_interval.Low.n,
			    bern_interval.Low.t,
			    bern_interval.Low.orders,
			    bern_interval.Low.X,
			    x))
	
    end
    println("Upper polys")
    show_matrices && @show bern_interval.Up.coeffs
    for i in axes(eachrow(bern_interval.Up.coeffs),1)

	println("$i:", imp_to_monomon(
			    NP.get_poly(i,bern_interval.Up),
			    bern_interval.Up.n,
			    bern_interval.Up.t,
			    bern_interval.Up.orders,
			    bern_interval.Up.X,
			    x))
	
    end

end
function multi_to_list(multi::NP.MultiBernsteinImp, n, X, x)
    res = []
    start = 1
    for i = 1:(length(multi.t))
        push!(
            res,
            imp_to_monomon(
                multi.coefficient_matrix[start:(start+multi.t[i]*n-1), 1:maximum(multi.orders)+1],
                n,
                multi.t[i],
                multi.orders,
                X,
                x,
            ),
        )
        start += multi.t[i] * n
    end
    return res

end
function coeff_dict(p)
    Dict(m => c for (m, c) in zip(monomials(p), coefficients(p)))
end

function poly_diff(p, q; tol = 1e-10)
    vp = variables(p)
    vq = variables(q)

    #println("p: $p, vp: $vp")
    #println("q: $q, vq: $vq")
    @assert length(vp) == length(vq) "Polynomials must have same number of variables"

    for i in eachindex(vp)
        q = subs(q, vq[i] => vp[i])
    end

    dp = coeff_dict(p)
    dq = coeff_dict(q)

    keys_all = union(keys(dp), keys(dq))

    diffs = []

    for k in keys_all
        cp = get(dp, k, 0.0)
        cq = get(dq, k, 0.0)

        if abs(cp - cq) > tol
            push!(diffs, (monomial = k, p = cp, q = cq, err = abs(cp - cq)))
        end
    end

    return diffs
end
function approx_equal(p, q; tol = 1e-10)
    isempty(poly_diff(p, q; tol = tol))
end
function test_poly_equal(p, q; tol = 1e-5)
    diffs = poly_diff(p, q; tol = tol)

    @test isempty(diffs) || error(
        "Polynomial mismatch:\n" *
        join(["$(d.monomial): p=$(d.p), q=$(d.q), err=$(d.err)" for d in diffs], "\n"),
    )
end
