using NNPoly, NeuralVerification
NV = NeuralVerification
NP = NNPoly
include("helper.jl")

for _ in 1:100
    n = rand(1:3)
    @polyvar x[1:n]
    lbs = fill(0.0, n) #round.( rand(n) .* 200 .- 100, digits=4)
    ubs = fill(1.0, n) #round.(rand(n) .* 200 .- 100, digits=4)

    lbs = [min(a, b) for (a, b) in zip(lbs, ubs)]
    ubs = [max(a, b) for (a, b) in zip(lbs, ubs)]
    #print(lbs)
    #print(ubs)

    for (i, lower) in enumerate(lbs)
        if ubs[i] == lower
            ubs[i] += 0.1
        end
    end
    X = Hyperrectangle(low = lbs, high = ubs)
    #println("X: $X")
    #println("W: $W")
    #println("b: $b")

    function random_polynomial(nvars, degrees, nterms)
        # create variables x1, x2, ..., xn

        p = 0

        for _ = 1:nterms
    	coeff = rand() * 20 - 10   # random coefficient in [-10,10]

    	# random monomial
    	monomial = prod(x[i]^rand(0:degrees[i]) for i = 1:nvars)

    	p += coeff * monomial
        end

        return p
    end

    degrees = [rand(1:3, n) for _ in 1:n]

    polys = [random_polynomial(n, degrees[i], rand(1:(n+maximum(degrees[i])))) for i in 1:n]
    #@show polys

    bern_polys = [NP.make_polynomial(poly, degrees[i], X) for (i,poly) in enumerate(polys)]

    multi_polys = NP.build_multi(bern_polys)

    for _ in 1:50
        a = rand(n) .* 200 .- 100
        b = rand(n) .* 200 .- 100
        c = rand(n) .* 200 .- 100
        res = a .* (polys).^2  .+ b .* polys .+ c
        #@show polys
        #@show [imp_to_monomon(p,x) for p in bern_polys]
        #@show multi_polys

        #print_multi_bernstein_imp(multi_polys,n,X)
        quad_prop = NP.quadratic_propagation(a,b,c,multi_polys)
        #print_multi_bernstein_imp(quad_prop,n,X)

        for (poly, bern_poly) in zip(res, multi_to_list(quad_prop, n, X, x))
            #println("checking $poly, $bern_poly")
            test_poly_equal(poly, bern_poly)
        end
    end
end 