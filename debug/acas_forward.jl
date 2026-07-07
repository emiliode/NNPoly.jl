using NNPoly, NeuralVerification, LazySets, Flux, DynamicPolynomials, CSV
const NP = NNPoly 
const NV = NeuralVerification

function run()
	
    solver = NP.BernSym()
    logfile = "./debug/acas_forward_all_layers.jld2"
    dir = "../vnncomp2022_benchmarks/benchmarks/acasxu"
    loss_fun = NP.bounds_loss_violation_stop
    f = CSV.File(string(dir, "/instances.csv"), header = false)

    # need y history to get access to final loss values

    force_gc = true
    n = length(f)
    networks = String[]
    properties = String[]
    y_starts = zeros(n)
    ys = zeros(n)
    y_hists = []
    t_hists = []
    lower_bounds = []
    upper_bounds = []
    t_hists = []
    #all_steps = zeros(Integer, n)
    times = zeros(n)

    old_netpath = nothing
    net = nothing
    net_original = nothing
    # net_npi = nothing
    n_in = 0
    n_out = 0

    cnt = 0

    for (i, instance) in enumerate(f)
        netpath, propertypath, time_limit = instance

        if netpath != old_netpath
            println("-- loading network ", netpath)
            net_original =
                NP.onnx2CROWNNetwork(solver, string(dir, "/", netpath), dtype = Float64)
            # net = read_onnx_network(string(dir, "/", netpath), dtype=Float64)
            old_netpath = netpath

            #net_npi = NV.NetworkNegPosIdx(net)

            n_in = size(net_original.layers[1].weights, 2)
            n_out = length(net_original.layers[end].bias)
            # n_out = NV.n_nodes(net.layers[end])
        end

        rv = NP.read_vnnlib_simple(string(dir, "/", propertypath), n_in, n_out)
        specs = NP.generate_specs(rv)
        input_set, output_set = specs[1]  # for now just use one set (we only care about the input set anyways here)

        # merge output property constraints into the last layer
        net = NP.merge_spec_output_layer(net_original, output_set)

        println("\n### Property ", propertypath, " ###\n")
        #println("--- initial α ---")
        #s = initialize_symbolic_domain(solver, net_npi, input_set)
        #α0 = initialize_params(solver, net_npi, 2, s)
        #y_start = propagate(solver, net_npi, s, α0; printing=true)

        #println("--- optimisation ---")
        println("input_set: $input_set")
	start_time = time()
	input = NP.init_combined_bernstein_interval(input_set)
	out = NP.forward_network(solver,net,input)
	lbs ,_ ,_,ubs = NP.bounds(out)
	loss  = loss_fun(lbs,ubs)
	res  = (t_hist= [time() - start_time], y_hist=[loss])
	end_time = time() -start_time

	println("\ttime = ", end_time)
        println("--- optimised α ---")
        #propagate(solver, net_npi, s, α₁; printing=true)
        println("\tlbs = ", lbs[end])
        println("\tubs = ", ubs[end])

        push!(networks, netpath)
        push!(properties, propertypath)
	times[i] = end_time
        y_starts[i] = res.y_hist[1]
	push!(y_hists,  res.y_hist)
        ys[i] = res.y_hist[end]
	push!(t_hists, res.t_hist)
	push!(lower_bounds, lbs)
	push!(upper_bounds, ubs)


        # also backup, if sth goes wrong later on
        if !isnothing(logfile)
            NP.save(
                logfile,
                "properties",
                properties,
                "times",
                times,
                "y_starts",
                y_starts,
                "ys",
                ys,
                "y_hists",
                y_hists,
                "t_hists",
                t_hists,
		"lbs",
		lower_bounds,
		"ubs",
		upper_bounds,
            )
        end

        if force_gc
            # force garbage collection 
            GC.gc()
        end
    end

    #=if !isnothing(logfile)
        open(logfile, "w") do f
            println(f, "network,property,result,time,steps")
            [println(f, string(network, ", ", property, ", ", result, ", ", time, ", ", steps))
                    for (network, property, result, time, steps) in zip(networks, properties, results, times, all_steps)]
        end
    end=#

    println("saving results ...")
    if !isnothing(logfile)
        NP.save(
            logfile,
            "properties",
            properties,
            "times",
            times,
            "y_starts",
            y_starts,
            "ys",
            ys,
            "y_hists",
            y_hists,
            "t_hists",
            t_hists,
	    "lbs",
	    lower_bounds,
	    "ubs",
	    upper_bounds,
        )
    end

end

run()
