using NNPoly, DynamicPolynomials, LazySets, Test, NeuralVerification
NP = NNPoly
NV = NeuralVerification
include("helper.jl")

function test()
    # input dimension
    max_error = 0
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
	multi_combined = NP.make_polynomial(polys, degrees, X)
	lbs_old, ubs_old =NP.bounds(multi,X;use_shortcut=true) #lbs_new, ubs_new, lbs_new_1, ubs_new_1 = NP.bounds(multi)	
	lbs_comb, ubs_comb = NP.bounds(multi_combined,X;use_shortcut=true)
	for _ in 1:100
	    p = point_in(X)
	    p_out = [ poly(p) for poly in polys  ]

	    for i in eachindex(lbs_old)
		error = 0
		if (p_out[i] < lbs_old[i]) 
		    error = abs((p_out[i] - lbs_old[i])/lbs_old[i])
		elseif (p_out[i] > ubs_old[i])
		    error = abs((p_out[i] - ubs_old[i])/ubs_old[i])
		end
		if error > max_error 
		    max_error = error
		end
		@assert lbs_old[i] == lbs_comb[i]
		@assert ubs_old[i] == ubs_comb[i]


		#@assert lbs_old[i] <= p_out[i] <= ubs_old[i] "Failed with $i "
		#@assert lbs_new_1[i] <= p_out[i] <= ubs_new_1[i] "Failed with $(lbs_new_1[i]) <= $(p_out[i]) <= $(ubs_new_1[i]) "
		#@assert lbs_new[i] <= p_out[i] <= ubs_new[i] "Failed with $(lbs_new[i]) <= $(p_out[i]) <= $(ubs_new[i]) "
	    end
	end
 

        #println("res_ub_polys: $res_ub_polys")
        #println("res_lb_polys: $res_lb_polys")



    end
    println("max error: $max_error")
end

# ca. 5.5 seconds
function run_test()
    @time test()
end

run_test()
