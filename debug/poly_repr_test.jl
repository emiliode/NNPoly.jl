using Test, NNPoly, DynamicPolynomials, BernsteinExpansions, LazySets, SpecialPolynomials
NP = NNPoly

function create_poly()
    @polyvar x y
    polynom = x^3 * y^2 - 30*x*y
    imp = NP.make_polynomial(polynom, [3, 2], Hyperrectangle(low = [1, 2], high = [2, 4]))
    @test NP.evaluate(imp, [1, 3.5]) == polynom(x => 1, y=>3.5)


    @test NP.dense(imp) == [[-174.0, -319.0, -580.0], [0, 0, 0], [0, 0, 0]]

    #NP.scalar_mul(imp, 2)

    #scaled_poly = 2*(x^3 * y^2 - 30*x*y)
    #@test imp.coefficient_matrix == NP.make_polynomial(scaled_poly, [3,2], Hyperrectangle(low=[1,2],high=[2,4])).coefficient_matrix


    #polynom = x^2  
    #imp =  NP.make_polynomial(polynom, [2], Hyperrectangle(low=[2],high=[4]))
    #@test NP.eval(imp,[2.5]) == polynom(x => 2.5 )
end

# uses bern_implicit to get bernstein basis vectors and ...
function eval_dense(bern_implicit::NP.BernsteinPolynomialImp, bern_dense, point)
    res = 0
    for I in CartesianIndices(bern_dense)
        #print("I=$I")
        prod = bern_dense[I]
        for i = 1:bern_implicit.n

            #println("\ti=$i")
            basis = bern_implicit.bernstein_basis[i][I[i]]
            #println("\tbasis=$basis")
            prod*=basis(variables(basis)[1] => point[i])
        end
        res += prod
    end
    return res
end

@testset "Some tests" begin
    @test 1 + 1 == 2

    #println(NP.bernstein_basis(3,-1,1))
    #create_poly()
end
@testset "DenseRepr" begin
    #@polyvar x y 
    #poly = x^3 * y^2 - 30*x*y 
    @polyvar x y
    poly = x^3 * y^2
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
    println(bern_dense)
    @test eval_dense(bern_implicit, bern_dense, [1.5, 2.5]) == poly(x => 1.5, y=>2.5)
end
