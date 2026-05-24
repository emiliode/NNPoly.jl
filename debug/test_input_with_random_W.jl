using NNPoly, DynamicPolynomials, LazySets, Test
NP = NNPoly 
include("helper.jl")

for _ in 1:100
    n = rand(1:8)

    lbs = rand(n) .* 200 .- 100
    ubs = rand(n) .* 200 .- 100


    lbs = [min(a,b) for (a,b) in zip(lbs,ubs)]
    ubs = [max(a,b) for (a,b) in zip(lbs,ubs)]
    # check that we don't have empty intervals:

    for (i,lower) in enumerate(lbs)
        if ubs[i] == lower 
            ubs[i] += 0.1
        end
    end


    X = Hyperrectangle(low=lbs,high=ubs)

    println("lbs: $lbs")
    println("ubs: $ubs")

    W_rows = rand(1:10)
    W = rand(W_rows,n) .* 200 .-100 
    b = rand(W_rows) .*200 .-100

    #set a random row to zero 20% of the time: 
    if (rand() >= 0.8)
        zero_row = rand(1:W_rows)
        W[zero_row,:] .= 0
    end

    println("W: $W")
    println("b: $b")
    bern_interval = NP.init_bernstein_interval(X) 

    @polyvar x[1:n]  
    polynom_up = [imp_to_monomon(bern_poly,x) for bern_poly in bern_interval.Up ] 
    polynom_up = [subs(p,variables(p)=> variables(polynom_up[1])) for p in polynom_up]
    polynom_low = [imp_to_monomon(bern_poly,x) for bern_poly in bern_interval.Low ] 
    polynom_low = [subs(p,variables(p)=> variables(polynom_low[1])) for p in polynom_low]
    println("polynom_up: $polynom_up")
    println("polynom_low: $polynom_low")
    mapped_poly_up = (min.(W,0) * polynom_up + max.(W,0)* polynom_low ) + b 
    mapped_poly_low = (min.(W,0) * polynom_low + max.(W,0)* polynom_up) + b 

    println("mapped_poly_up: $mapped_poly_up")
    println("mapped_poly_low: $mapped_poly_low")

    res = NP.interval_map(min.(W,0), max.(W,0),bern_interval,b)


    for ((bern_low, bern_up), (poly_low,poly_up)) in zip(zip(res.Low, res.Up), zip(mapped_poly_low,mapped_poly_up))
        println("testing lower")
        println("poly_low: $poly_low")
        println("bern_low: $bern_low")
        test_poly_equal(imp_to_monomon(bern_low,x),poly_low)
        println("testing upper")
        test_poly_equal(imp_to_monomon(bern_up,x),poly_up)
    end
end
