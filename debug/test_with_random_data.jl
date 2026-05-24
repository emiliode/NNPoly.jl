using NNPoly, DynamicPolynomials, LazySets, Test
NP = NNPoly 
include("helper.jl")

# input dimension
for _ in 1:100
    n = rand(1:10)

    @polyvar x[1:n]  
    lbs = fill(0.0,n) #round.( rand(n) .* 200 .- 100, digits=4)
    ubs = fill(1.0,n) #round.(rand(n) .* 200 .- 100, digits=4)

    lbs = [min(a,b) for (a,b) in zip(lbs,ubs)]
    ubs = [max(a,b) for (a,b) in zip(lbs,ubs)]
    print(lbs)
    print(ubs)

    for (i,lower) in enumerate(lbs)
        if ubs[i] == lower 
            ubs[i] += 0.1
        end
    end
    X = Hyperrectangle(low=lbs,high=ubs)
    W_rows = rand(1:5)
    W = rand(W_rows,n) .* 200 .-100
    b = rand(W_rows) .*200 .-100
    println("X: $X")
    println("W: $W")
    println("b: $b")

    function random_polynomial(nvars, degrees, nterms)
        # create variables x1, x2, ..., xn

        p = 0

        for _ in 1:nterms
            coeff = rand() * 20 - 10   # random coefficient in [-10,10]

            # random monomial
            monomial = prod(x[i]^rand(0:degrees[i]) for i in 1:nvars)

            p += coeff * monomial
        end

        return p
    end

    degrees = rand(1:3,n)

    ub_polys = [random_polynomial(n,degrees,rand(1:n+maximum(degrees))) for _ in 1:n] 
    lb_polys = [random_polynomial(n,degrees,rand(1:n+maximum(degrees))) for _ in 1:n] 
    println("ub_polys: $ub_polys")
    println("lb_polys: $lb_polys")

    W⁻ = min.(W,0)
    W⁺ = max.(W,0)

    res_lb_polys = (W⁻ * ub_polys + W⁺ * lb_polys) + b
    res_ub_polys = (W⁻ * lb_polys + W⁺ * ub_polys) + b

    ub_bern_polys = [NP.make_polynomial(poly,degrees,X) for poly in ub_polys]
    lb_bern_polys = [NP.make_polynomial(poly,degrees,X) for poly in lb_polys]
    println("ub_bern_polys: $ub_bern_polys")
    println("lb_bern_polys: $lb_bern_polys")

    println("res_ub_polys: $res_ub_polys")
    println("res_lb_polys: $res_lb_polys")


    bern_interval = NP.BernsteinInterval(lb_bern_polys,ub_bern_polys)


    res = NP.interval_map(min.(W,0),max.(W,0),bern_interval,b)

    # test upper polys equal
    for (poly, bern_poly) in zip(res_ub_polys, res.Up)
        println("checking $poly, $(bern_poly.coefficient_matrix)")
        println("$bern_poly")
        test_poly_equal(poly, imp_to_monomon(bern_poly,x))
    end

    # test lower polys equal
    for (poly, bern_poly) in zip(res_lb_polys, res.Low)
        println("checking $poly, $(bern_poly.coefficient_matrix)")
        test_poly_equal(poly, imp_to_monomon(bern_poly,x))
    end
end

#p = -26.820124416573066 - 281.62263391129557*x[2]^2 + 359.7204995618044*x[2]^2*x[3] - 219.85444381277304*x[1]*x[2]^2 - 113.78361835590215*x[1]^2*x[2] + 492.93717165474897*x[1]*x[2]^2*x[3] - 4.598034953701051*x[1]^3*x[3] + 414.2962745671388*x[1]^3*x[2]^2 - 54.97044029675252*x[1]^3*x[2]^2*x[3]
#
#
#X = Hyperrectangle([-54.7564, 38.9628, 28.6647], [0.05000000000000426, 0.04999999999999716, 15.6981])
#bern = NP.make_polynomial(p,[3,3,3],X)
#test_poly_equal(p, imp_to_monomon(bern,x))
@polyvar x[1:3]
bern=NNPoly.BernsteinPolynomialImp([0.0 0.0 -219.27878675964533 0.0; 1.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0; -333.3815572817581 -333.3815572817581 -333.3815572817581 0.0; 0.0 0.0 0.0 1.0; 1.0 1.0 0.0 0.0; 0.0 0.0 -329.49076118355896 0.0; 0.0 0.0 0.0 1.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0; 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.3333333333333333 1.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 1.0 0.0 0.0; 0.0 -121.28700758974645 -242.5740151794929 0.0; 1.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0; 354.1172359786037 354.1172359786037 354.1172359786037 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0; 0.0 1.0 0.0 0.0; 0.0 0.0 288.73146409952017 0.0; 1.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0; 0.0 0.0 -68.97272218639672 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0; 0.0 1.0 0.0 0.0; 709.0951774515536 709.0951774515536 709.0951774515536 0.0; 1.0 1.0 1.0 1.0; 0.0 1.0 0.0 0.0; 0.0 0.0 -184.3952398392934 0.0; 1.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0; 90.53052301352994 90.53052301352994 90.53052301352994 0.0; 0.0 0.0 0.0 1.0; 0.0 1.0 0.0 0.0; -99.36568831473633 -99.36568831473633 -99.36568831473633 -0.0; 1.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0], 3, 16, Hyperrectangle([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), [2, 3, 1])
println(imp_to_monomon(bern,x))