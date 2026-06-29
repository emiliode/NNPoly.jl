using NNPoly, DynamicPolynomials, LazySets, Test, NeuralVerification
NP = NNPoly
NV = NeuralVerification
include("helper.jl")

for _ = 1:100
    n = rand(1:8)

    lbs = rand(n) .* 200 .- 100
    ubs = rand(n) .* 200 .- 100


    lbs = [min(a, b) for (a, b) in zip(lbs, ubs)]
    ubs = [max(a, b) for (a, b) in zip(lbs, ubs)]
    # check that we don't have empty intervals:

    for (i, lower) in enumerate(lbs)
        if ubs[i] == lower
            ubs[i] += 0.1
        end
    end


    X = Hyperrectangle(low = lbs, high = ubs)

    println("lbs: $lbs")
    println("ubs: $ubs")

    W_rows = rand(1:10)
    W = rand(W_rows, n) .* 200 .- 100
    b = rand(W_rows) .* 200 .- 100

    #set a random row to zero 20% of the time: 
    if (rand() >= 0.8)
        zero_row = rand(1:W_rows)
        W[zero_row, :] .= 0
    end

    println("W: $W")
    println("b: $b")
    bern_interval = NP.init_bernstein_interval(X)
    bern_com_interval = NP.init_combined_bernstein_interval(X)

    @polyvar x[1:n]
    polynom_up = multi_to_list(bern_interval.Up, bern_interval.n, bern_interval.X, x)
    polynom_up = [subs(p, variables(p) => variables(polynom_up[1])) for p in polynom_up]
    polynom_low = multi_to_list(bern_interval.Low, bern_interval.n, bern_interval.X, x)
    polynom_low = [subs(p, variables(p) => variables(polynom_low[1])) for p in polynom_low]


    println("polynom_up: $polynom_up")
    println("polynom_low: $polynom_low")
    mapped_poly_up = (min.(W, 0) * polynom_up + max.(W, 0) * polynom_low) + b
    mapped_poly_low = (min.(W, 0) * polynom_low + max.(W, 0) * polynom_up) + b

    println("mapped_poly_up: $mapped_poly_up")
    println("mapped_poly_low: $mapped_poly_low")

    res = NP.interval_map(min.(W, 0), max.(W, 0), bern_interval, b)

    res_com = NP.interval_map(min.(W,0),max.(W,0),bern_com_interval,b)

    res_bern_low = multi_to_list(res.Low, res.n, res.X, x)
    res_bern_up = multi_to_list(res.Up, res.n, res.X, x)

    for i in eachindex(mapped_poly_up)
        println("testing lower")
	println("poly_low: $(mapped_poly_low[i])")
	println("bern_low: $(res_bern_low[i])")
	test_poly_equal(mapped_poly_low[i], res_bern_low[i])
        println("testing upper")
	test_poly_equal(res_bern_up[i], mapped_poly_up[i])
        println("testing bern combined")
	com_up = imp_to_monomon(NP.get_poly(i,res_com.Up),res_com.Up.n,res_com.Up.t,res_com.Up.orders,res_com.Up.X,x)
	com_low = imp_to_monomon(NP.get_poly(i,res_com.Low),res_com.Low.n,res_com.Low.t,res_com.Low.orders,res_com.Low.X,x)
	@show com_up
	@show mapped_poly_up[i]
	test_poly_equal(com_up, mapped_poly_up[i])
	@show com_low
	@show mapped_poly_low[i]
	test_poly_equal(com_low, mapped_poly_low[i])
    end
end
