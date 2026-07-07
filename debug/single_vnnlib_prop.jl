using NNPoly, NeuralVerification, Flux, LazySets
NP = NNPoly
NV = NeuralVerification


function run()
    #netpath, propertypath, time_limit = instance
    solver = NP.PolyCROWNBern()
    netpath = "onnx/mnist-net_256x6.onnx"
    propertypath = "vnnlib/prop_0_spiral_2.vnnlib" 
    time_limit= 300

    loss_fun = NP.bounds_loss
    params = NP.OptimisationParams(
        y_stop =  0.0,
        print_freq = 1,
        n_steps = 1000,
        timeout = 300)
    dir = "./eval/mnist_fc"

    println("-- loading network ", netpath)
    #net_original =
    #	NP.onnx2CROWNNetwork(solver, string(dir, "/", netpath), dtype = Float64)
    #    # net = read_onnx_network(string(dir, "/", netpath), dtype=Float64)
    #    #net_npi = NV.NetworkNegPosIdx(net)
    #
    #n_in = size(net_original.layers[1].weights, 2)
    #n_out = length(net_original.layers[end].bias)
    #    # n_out = NV.n_nodes(net.layers[end])
    #
    #
    #rv = NP.read_vnnlib_simple(string(dir, "/", propertypath), n_in, n_out)
    #specs = NP.generate_specs(rv)
    #input_set, output_set = specs[1]  # for now just use one set (we only care about the input set anyways here)

    input_set = Hyperrectangle(low=[0.], high=[1.])
    #net = NP.merge_spec_output_layer(net_original, output_set)
    n_in = 1 
    n_out = 1 
    net = W1 = reshape([1.; 1], 2, 1)  # need matrix
    b1 = [0.5, -0.5]
    W2 = [1 -1.]
    b2 = [0.]

    L1 = NP.CROWNLayer(W1, b1, NV.ReLU(), zeros(2, 2, 2))
    L2 = NP.CROWNLayer(W2, b2, NV.Id(), similar(b2, 0))
    net = Chain(L1, L2)

    println("\n### Property ", propertypath, " ###\n")
    #println("--- initial α ---")
    #s = initialize_symbolic_domain(solver, net_npi, input_set)
    #α0 = initialize_params(solver, net_npi, 2, s)
    #y_start = propagate(solver, net_npi, s, α0; printing=true)

    #println("--- optimisation ---")
    println("input_set: $input_set")
    time_required = @elapsed res, lbs, ubs = NP.optimise_bounds(
        solver,
        net,
        input_set,
        params = params,
        loss_fun = loss_fun; 
        use_combined_repr = true
    )
    #α₁ = res.x_opt

    println("\ttime = ", time_required)
    println("--- optimised α ---")
    #propagate(solver, net_npi, s, α₁; printing=true)
    println("\tlbs = ", lbs[end])
    println("\tubs = ", ubs[end])
    println("\tres = ",res)

end

run()
@profview run()