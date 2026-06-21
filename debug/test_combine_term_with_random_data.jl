using NNPoly, DynamicPolynomials, LazySets, Test, NeuralVerification
NP = NNPoly
NV = NeuralVerification
include("helper.jl")

function test()
    # input dimension
    saved_terms = 0
    for _ = 1:20
        n = rand(1:10)

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

	degree= rand(1:3)
	degrees = fill(degree,n) #rand(1:3, n)

        polys = [random_polynomial(n, degrees, rand(1:(n+maximum(degrees)))) for _ = 1:n]
        #println("ub_polys: $ub_polys")
        #println("lb_polys: $lb_polys")

        bern_polys = [NP.make_polynomial(poly, degrees, X) for poly in polys]

        multi = NP.build_multi(bern_polys)

	lb_before,ub_before  = NP.bounds(multi)
	

	multi_combined = NP.combine_terms(multi)

	lb_after,ub_after  = NP.bounds(multi_combined)
	# check bounds 
	for i in eachindex(lb_before)
	    @test isapprox(lb_before[i],lb_after[i])
	    @test isapprox(ub_before[i],ub_after[i])
	end

	cur_saved_terms = maximum( multi.t .- multi_combined.t) 
	if cur_saved_terms  > saved_terms 
	    saved_terms = cur_saved_terms 
	end
	@polyvar x[1:n]
	for (poly, combined_poly) in zip(polys, multi_to_list(multi_combined,n, X,x))
	    test_poly_equal(combined_poly,poly) 
	end
	#@test multi_combined.coefficient_matrix == reduce(vcat,combined_polys)

        #println("res_ub_polys: $res_ub_polys")
        #println("res_lb_polys: $res_lb_polys")



    end
    println("max saved_terms: $saved_terms")
end

# ca. 5.5 seconds
function run_test()
    @time test()
end

run_test()
