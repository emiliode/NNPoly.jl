using NNPoly, NeuralVerification, LazySets, Flux, DynamicPolynomials
const NP = NNPoly 
const NV = NeuralVerification

include("helper.jl")

"""
Sample a random point uniformly from a hyperrectangle.
"""


UPPER_VALUE_LIM = 1000
LOWER_VALUE_LIM = -1000
RANGE =  UPPER_VALUE_LIM - LOWER_VALUE_LIM

function random_layer(input_vars::Int64, output_vars::Int64, idx::Int64)
    W = rand(output_vars, input_vars) .* RANGE .- (RANGE/2)
    b = rand(output_vars) .* RANGE .- (RANGE/2)
    
    return NV.LayerNegPosIdx(W,b,min.(W,0),max.(W,0),NV.ReLU(),idx)
end

function random_net(n::Int)
    num_layers = rand(1:3)
    layers  = Vector{NV.LayerNegPosIdx}(undef, num_layers)
    cur_input_vars = n
    cur_output_vars = rand(1:4)
    for i in eachindex(layers)
        layers[i] = random_layer(cur_input_vars, cur_output_vars,i) 
        cur_input_vars =  cur_output_vars 
        cur_output_vars = rand(1:4)
    end
    layers[end] = NV.LayerNegPosIdx(
        layers[end].weights,
        layers[end].bias,
        layers[end].W_neg,
        layers[end].W_pos,
        NV.Id(),              
        layers[end].index
    )
    return Chain(layers)
end

n = rand(1:3)
net = random_net(n)
#W1 = reshape([1.; 1], 2, 1)  # need matrix
#W1 = reshape([1.; 1], 2, 1)  # need matrix
#b1 = [0.5, -0.5]
#W1= [1. 1; 1. 1]
#
#W3 = [1 -1.]
#b2 = [0.]
#L1 = NV.LayerNegPosIdx(W1,b1, min.(W1,0), max.(W1,0),NV.ReLU(),1)
#L2 = NV.LayerNegPosIdx(W2,b1, min.(W2,0), max.(W2,0),NV.ReLU(),2)
#L2 = NV.LayerNegPosIdx(W3,b2, min.(W3,0), max.(W3,0),NV.Id(),2)
#
#net = Chain(L1,L2)
input_set = Hyperrectangle(low=fill(-1,n), high=fill(1,n) )
input = NP.init_bernstein_interval(input_set)

solver = NP.BernSym(common_generators = false, init=true)

out = NP.forward_network(solver, net,input)
#crown_out = NP.forward_network(crown_solver, net, NP.initialize_symbolic_domain(crown_solver,net,input_set))

@polyvar x[1:n]
upper_polys = multi_to_list(out.Up,n,input_set,x)
lower_polys = multi_to_list(out.Low,n,input_set,x)
println(upper_polys)
println(lower_polys)
#check if bounds are correct by testing them for random points.
max_error = 0
_ , upper_bounds = NP.bounds(out.Up)
lower_bounds , _ = NP.bounds(out.Low)
for _ in 1:100
    p = point_in(input_set)
    p_out = forward(net, p)
    #upper_bounds = [ upper_poly(p) for upper_poly  in upper_polys]
    #lower_bounds = [  lower_poly == 0 ? 0 : lower_poly(p) for lower_poly in lower_polys ]

    for ((lb,ub),p_i) in zip(zip(lower_bounds, upper_bounds),p_out)
	error = 0
	if (p_i < lb) 
	    error = abs((p_i - lb)/lb)
	elseif (p_i > ub)
	    error = abs((p_i - ub)/ub)
	end
	if error > max_error 
	    global max_error = error
	end
        #if lb > p_i || ub < p_i
        #   throw("FAILED: $lb > $p_i || $ub < $p_i") 
        #end
    end
end
println("max error: $max_error")
