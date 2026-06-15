
using NNPoly, NeuralVerification, LazySets, Flux, DynamicPolynomials
const NP = NNPoly 
const NV = NeuralVerification

include("helper.jl")

UPPER_VALUE_LIM = 100
LOWER_VALUE_LIM = -100
RANGE =  UPPER_VALUE_LIM - LOWER_VALUE_LIM
function random_crown_layer(input_vars::Int64, output_vars::Int64, d::Int64)
    W = round.( rand(output_vars, input_vars) .* RANGE .- (RANGE/2); digits=1)
    b = round.( rand(output_vars) .* RANGE .- (RANGE/2); digits=1)
    
    if d == 1
        α = similar(b)
    else
        # n_neurons × degree × 2 (params for lower and upper bound)
        α = similar(b, length(b), d, 2)
    end
    @assert output_vars == length(b)
    return NP.CROWNLayer(W,b,NV.ReLU(),α) # (number of neurons, degree, 2 (for lower and upper))
end

function random_crown_net(n::Int)
    num_layers = rand(2:3)
    layers  = Vector{NP.CROWNLayer}(undef, num_layers)
    cur_input_vars = n
    cur_output_vars = rand(1:4)
    for i in eachindex(layers)
        degree = i == 1 ? 2 : 1
        @show degree
        layers[i] = random_crown_layer(cur_input_vars, cur_output_vars,degree) 
        cur_input_vars =  cur_output_vars 
        cur_output_vars = rand(1:4)
    end
    layers[end] = NP.CROWNLayer(
        layers[end].weights,
        layers[end].bias,
        NV.Id(),              
        #similar(layers[end].bias)
        zeros(length(layers[end].bias),1)
    )
    return Chain(layers)
end
#n = #rand(1:3)
#W1 = reshape([1.; 1], 2, 1)  # need matrix
#b1 = [0.5, -0.5]
#W2 = [1 -1.]
#b2 = [0.]
#
#L1 = NP.CROWNLayer(W1, b1, NV.ReLU(), zeros(2, 2, 2))
#L2 = NP.CROWNLayer(W2, b2, NV.Id(), similar(b2, 0))
function run_tests(num_tests)
    solver = NP.PolyCROWNBern()
    run_errors=[]
    max_error = 0 
    max_net= Chain()
    max_input_set = Hyperrectangle(low=[0],high=[0])
    max_total = 0 
    for test_id in 1:num_tests
        n = rand(1:3)
        net_pcrown = random_crown_net(n) 
        input_set = Hyperrectangle(low=fill(0,n), high=fill(1,n) )
        input = NP.init_bernstein_interval(input_set)

        @show net_pcrown
        ŝ, lbs, ubs = NP.initialize_params_bounds(solver,net_pcrown,2,input)
        s_poly = NP.forward_act_stub(
                solver.poly_solver,
                net_pcrown[1],
                ŝ,
                lbs[1],
                ubs[1],
            )
        s_crown = NV.forward_network(
            solver.lin_solver,
            net_pcrown[2:end],
            s_poly,
            lbs[2:end],
            ubs[2:end],
        )

        ll, lu = NP.bounds(s_crown.Λ, s_crown.λ, s_poly;use_shortcut=solver.poly_solver.use_shortcut)
        ul, uu = NP.bounds(s_crown.Γ, s_crown.γ, s_poly;use_shortcut=solver.poly_solver.use_shortcut)
	error_total = 0
	wrong_bounds  =0
        for _ in 1:1000
            p = point_in(input_set)
            p_out = forward(net_pcrown, p)
            #upper_bounds = [ upper_poly(p) for upper_poly  in upper_polys]
            #lower_bounds = [  lower_poly == 0 ? 0 : lower_poly(p) for lower_poly in lower_polys ]

	    error_point = 0
            for ((lb,ub),p_i) in zip(zip(ll, uu),p_out)
		#tol = 5e-2 * max(abs(lb), abs(ub), 1.0)
		error = 0
		total = 0
		if (p_i < lb) 
		    error = abs((p_i - lb)/lb)
		    total = abs(p_i -lb)
  println(p)
		elseif (p_i > ub)
		    error = abs((p_i - ub)/ub)
		    total = abs(p_i -ub)
  println(p)
		end
		if error > max_error 
		    max_error = error
		    max_total = total
            max_net=  net_pcrown
            max_input_set = input_set
		end
		error_point += error
		wrong_bounds += 1
                #if !(lb -tol <= p_i <= ub + tol)
                #    error("""
                #    PolyCROWN bound violation

                #    Network:
                #    $net_pcrown

                #    Input:
                #    $p

                #    Output:
                #    $p_out


                #    Bounds:
                #    ($ll, $uu)
                #    """)
                #end
            end
	    #println("point_error: $error_point")
	    error_total +=error_point
        end
	println("error total: $error_total, avg: $(error_total/wrong_bounds)")
	push!(run_errors, error_total/wrong_bounds)
    end
    @show run_errors
    println(" avg: $(sum(run_errors) / length(run_errors)), max: $max_error, $max_total" )
    @show max_input_set
    @show max_net
end 

@time run_tests(1) 
@time run_tests(100) 

#
# run_errors = Any[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9865499804197393e-17, 0.0, 0.0, 0.0, 0.0, -1.6658436713187937e-6, 0.0019131145079683352, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00015393317484960846, -0.00031287522327546225, 0.0, 0.0, 0.0009255401817660559, 0.0, 0.0, 0.0, 0.0, 1.4770429458053195e-5, 4.865696525961306e-18, 0.0, 5.8043747977165665e-5, 0.0, 0.0, 0.0, -0.002044024547791706, 0.0, 0.0005009773722766585, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00010451478934841566, 0.0, 0.0, 0.0, 0.0, 0.00028061907010559604, 3.2822149661894043e-9, 0.0, 0.0, -2.8790705643793692e-5, 0.0, 0.0, -7.858270674382285e-6, 0.0, 0.0, 0.0, -0.0029671269688073066, 0.0, 0.0, 0.0, -0.0010513261330916974, 0.0, 0.0012107282210702724, 0.0, -1.3031769706311328e-5, 0.0006066519645804667, 0.0, 0.00196808640542247, 0.0, -0.00012096810053750978, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0001345084942993008, 0.0, -0.0007673882672084806, 0.0, 0.0, 0.0, -3.922882955722081e-6, 0.000569023943097007, 0.0004469796291488863, 0.0, 0.0, 0.0, 0.0, 0.0, 2.018966343104726e-5, 0.0, -0.0017827359846235974]
 #avg: -7.709131592708373e-6
#
#
#
