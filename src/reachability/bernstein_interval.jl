using LazySets
struct BernsteinInterval{N,M,O,TN}
    Low::Vector{BernsteinPolynomialImp{N,M,O,TN}}    
    Up::Vector{BernsteinPolynomialImp{N,M,O,TN}}    

end
"""
Construct a bernstein Polynomial repr of x_i over the input domain X 
"""
function init_one_bernstein_interval(i, X::Hyperrectangle)
    coeff_mat = fill(1.0,dim(X),2)
    coeff_mat[i,:] = [low(X)[i],high(X)[i]]
    return BernsteinPolynomialImp(coeff_mat,dim(X),1,X,fill(1,dim(X)))
end

"""
Construct a BernsteinInterval over h by filling Low and Up with x 
"""
function  init_bernstein_interval(h::Hyperrectangle)
    n = dim(h)
    @polyvar x[1:n]
    print(typeof(x))
    polys = [
        init_one_bernstein_interval(i,h) for i in 1:n]
    println(typeof(polys[1]))
    return BernsteinInterval(polys,copy(polys))
end

function linear_map(mat, polys::Vector{BernsteinPolynomialImp{N,M,O,TN}}) where {N<:Number,M<:Integer,O<:Number,TN<:AbstractArray{O}}
    res_polys= similar(polys,size(mat,1))
    for (i,row) in enumerate(eachrow(mat))
        #println("row: $row")
        #println("polys: $polys")
        s_muls = scalar_mul(polys[1],row[1])
        #println("s_muls: $s_muls")
        lin_polys = [scalar_mul(poly,c) for (c,poly)in zip(row,polys)]
        #println("lin_polys: $lin_polys")
        res_polys[i] = foldl(add,lin_polys)
    end
    return res_polys
end

function add(polys_a::Vector{<:BernsteinPolynomialImp}, polys_b::Vector{<:BernsteinPolynomialImp}) 
    return [ add(poly_a,poly_b) for (poly_a,poly_b) in zip(polys_a,polys_b) ]
end

function translate(polys::Vector{<:BernsteinPolynomialImp}, bs::Vector{N}) where{N<:Number}
   return [translate(poly, b) for (poly,b) in zip(polys,bs)] 
end

"""
Interval map overapproximating an affine map Wx + b for x ∈ I 

W⁻ - (matrix) negative weights
W⁺ - (matrix) positive weights
I  - (BernsteinInterval) bernsteinInterval
b  - (vector) bias
"""
function interval_map(W⁻, W⁺, I::BernsteinInterval, b)
    new_low = add(linear_map(W⁻,I.Up), linear_map(W⁺,I.Low) )
    new_up = add(linear_map(W⁻,I.Low), linear_map(W⁺,I.Up) )
    new_low =  translate(new_low, b)
    new_up  = translate(new_up, b)
    return BernsteinInterval(new_low,new_up)
end

