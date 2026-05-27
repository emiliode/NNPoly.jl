
using Test, Plots, NNPoly, DynamicPolynomials, LazySets, NeuralVerification
NP = NNPoly
NV = NeuralVerification
include("helper.jl")




function plot()
    @polyvar x y
    p = x^2 + y^2 - x*y + 2x-y

    poly = x^3 * y^2 - 30*x*y
    # Basis degree = 3, [1,2]
    # ber_0 = (2-x)^3 
    # ber_1 = 3*(x-1)* (2-x)^2 
    # ber_2 = 3* (x-1)^2 * (2-x)
    # ber_3 = (x-1)^3
    # Basis degree 2, [2,4]
    # ber_0 = 1/4 *(4-x)^2
    # ber_1 = 2/4 *(x-2)*(4-x)
    # ber_2 = 1/4 *(x-2)^2
    #bern_implicit =  NP.make_polynomial(poly, [3,2], Hyperrectangle(low=[1,2],high=[2,4]))
    bern_implicit =
        NP.make_polynomial(poly, [3, 2], Hyperrectangle(low = [1, 2], high = [2, 4]))
    #@test all(isapprox.(NP.dense(bern_implicit), [-174.0  -319 -580; -228 -418 -760;-276 -506 -920; -312 -572 -1040]))
    #bern_dense = [-174.0 -319.0 -580.0; -228.0 -418.0 -760.0; -275.99999999999994 -505.99999999999994 -919.9999999999999; -312.0 -572.0 -1040.0]
    bern_dense = NP.dense(bern_implicit)
    println("bern_dense: ", bern_dense)
    #@test poly == imp_to_monomon(bern_implicit)
    monomial_from_dense = dense_to_monomon(bern_dense, bern_implicit)
    #@test poly == monomial_from_dense


    xs = range(1, 2; length = 200)
    ys = range(2, 4; length = 400)

    surface(
        range(1, 2; length = 50),
        range(2, 4; length = 10),
        (x, y) -> eval_dense(bern_implicit, bern_dense, [x, y]),
        xlabel = "x",
        ylabel = "y",
        zlabel = "dense",
    )

    surface(
        xs,
        ys,
        (xv, yv) -> poly(x=>xv, y=>yv),
        xlabel = "x",
        ylabel = "y",
        zlabel = "real ",
        reuse = false,
    )

end


@testset "multiplication test" begin
    @polyvar x y
    p1 = x*y + x^2*y + y^3
    p2 = x*y^2 + x*y

    bern_imp1 = NP.make_polynomial(p1, [2, 3], Hyperrectangle(low = [1, 2], high = [2, 4]))
    bern_imp2 = NP.make_polynomial(p2, [1, 2], Hyperrectangle(low = [1, 2], high = [2, 4]))

    multiplied = NP.multiply(bern_imp1, bern_imp2)
    #    println("real: $(p1*p2)")
    #    println("mul: $(imp_to_monomon(multiplied))")
    test_poly_equal((p1 * p2), imp_to_monomon(multiplied, [x,y]))
end

@testset "addition tests" begin
    @polyvar x y
    p1 = x*y
    p2 = x*y^2
    bern_imp1 = NP.make_polynomial(p1, [1, 1], Hyperrectangle(low = [1, 2], high = [2, 4]))
    bern_imp2 = NP.make_polynomial(p2, [1, 2], Hyperrectangle(low = [1, 2], high = [2, 4]))

    res = NP.add(bern_imp1, bern_imp2)

    test_poly_equal(p1+p2, imp_to_monomon(res, [x,y]))

    p3 = x*y^3 + x*y
    bern_imp3 = NP.make_polynomial(p3, [1, 3], Hyperrectangle(low = [1, 2], high = [2, 4]))
    res1 = NP.add(bern_imp1, bern_imp3)

    test_poly_equal(p1+p3, imp_to_monomon(res1, [x,y]))

end
@testset "const add test" begin
    @polyvar x y
    p1 = x*y
    bern_imp1 = NP.make_polynomial(p1, [1, 1], Hyperrectangle(low = [1, 2], high = [2, 4]))
    c=5

    res = NP.translate(bern_imp1, c)

    println(imp_to_monomon(res,[x,y]))
    test_poly_equal(p1 + c, imp_to_monomon(res,[x,y]))
end

@testset "propagate through linear" begin
    using NNPoly, LazySets, NeuralVerification
    NP = NNPoly
    NV = NeuralVerification
    include("helper.jl")
    W1 = [-1 1; -1 1;]
    b1 = [0.5, -0.5]

    L1 = NP.CROWNLayer(W1, b1, NV.Id(), zeros(2, 2, 2))

    input_set = Hyperrectangle(low = [-1, -1], high = [1, 1])

    bern_input_set = NP.init_bernstein_interval(input_set)
    println("low:")
    print_multi_bernstein_imp(bern_input_set.Low, bern_input_set.n, bern_input_set.X)
    println("high:")
    print_multi_bernstein_imp(bern_input_set.Up, bern_input_set.n, bern_input_set.X)

    scaled_bern_input_set = NP.BernsteinInterval(
        NP.scalar_mul(bern_input_set.Low, 2, 2),
        NP.scalar_mul(bern_input_set.Up, 2, 2),
        2,
        input_set,
    )
    translated = NP.translate(bern_input_set.Low, [5, 5])
    print_multi_bernstein_imp(translated, 2, input_set)
    println("*")
    print_multi_bernstein_imp(scaled_bern_input_set.Low, 2, input_set)
    print_multi_bernstein_imp(
        NP.multiply(translated, scaled_bern_input_set.Low),
        2,
        input_set,
    )


    W = [2 1; 1 2; 3 2]
    print_multi_bernstein_imp(
        NP.linear_map(W, bern_input_set.Low),
        bern_input_set.n,
        bern_input_set.X,
    )
    @polyvar x[1:2]

    out_bern = NP.interval_map(min.(L1.weights, 0), max.(L1.weights, 0), bern_input_set, b1)

    for (lower, upper) in zip(out_bern.Low, out_bern.Up)
        println("lower: ", imp_to_monomon(lower,x))
        println("upper: ", imp_to_monomon(upper,x))
    end


end
#TODO: testen mit zufällige Daten mit masks für edge cases 

function test_scalar_mul()
    @time for i = 2:100
        input_set = Hyperrectangle(low = fill(-1, 1000), high = fill(1, 1000))

        scalars = collect(i:(1000+i))
        bern_input_set = NP.init_bernstein_interval(input_set)

        new_low = NP.scalar_mul(bern_input_set.Low, bern_input_set.n, scalars)
        new_up = NP.scalar_mul(bern_input_set.Up, bern_input_set.n, scalars)
        bern_input_set =
            NP.BernsteinInterval(new_low, new_up, bern_input_set.n, bern_input_set.X)
    end
end
test_scalar_mul()
