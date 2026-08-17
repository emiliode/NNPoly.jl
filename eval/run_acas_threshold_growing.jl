using NNPoly
import NNPoly:  SmithBoundsOverapproximate, SmithBoundsMonomon,Overapproximate, Imp, CombinedImp, Dense, DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib, PolyCROWNBern, bounds_loss_violation_stop, violation_loss, bounds_loss

ACAS_PATH = "../vnncomp2022_benchmarks/benchmarks/acasxu"
loss_fun = bounds_loss_violation_stop
loss_fun_name = "bounds_loss_violation_stop"

THRESHOLD=50

use_shortcut = false
println("precompiling ...")
pcrown = PolyCROWNBern(use_memory_optimizations=true, bounds_method=SmithBoundsMonomon,interval_repr=CombinedImp , poly_layers = 1, threshold=THRESHOLD)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
    logfile = "./eval/acas_results_polycrownbern_final_mem_opt_combinedimp_overapproximate_opt_$THRESHOLD.jld2",
    max_properties = 1,
    print_freq = 1,
    n_steps = 1,
    save_history = true,
    save_times = true,
    timeout = 300,
    loss_fun = bounds_loss,
)
pcrown = PolyCROWNBern(use_memory_optimizations=true, bounds_method=SmithBoundsOverapproximate,interval_repr=CombinedImp , poly_layers = 1, threshold=THRESHOLD)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
    logfile = "./eval/acas_results_polycrownbern_final_mem_opt_combinedimp_smithboundsoverapproximate_opt_$THRESHOLD.jld2",
    max_properties = 1,
    print_freq = 1,
    n_steps = 1,
    save_history = true,
    save_times = true,
    timeout = 300,
    loss_fun = bounds_loss,
)
#STEPS = typemax(Int)
STEPS = 1
println("running experiments ...")
for THRESHOLD in 1:10:251
    pcrown = PolyCROWNBern(use_memory_optimizations=true, bounds_method=SmithBoundsMonomon,interval_repr=CombinedImp , poly_layers = 1, threshold=THRESHOLD)
    properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
        pcrown,
        ACAS_PATH,
        #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
        #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
        logfile = "./eval/acas_results_polycrownbern_final_mem_opt_combinedimp_smithboundsmonomon_opt_$THRESHOLD.jld2",
        max_properties = Inf,
        print_freq = 5,
        n_steps = STEPS,
        save_history = true,
        save_times = true,
        timeout = 300,
        loss_fun = bounds_loss,
    )
    pcrown = PolyCROWNBern(use_memory_optimizations=true, bounds_method=SmithBoundsOverapproximate,interval_repr=CombinedImp , poly_layers = 1, threshold=THRESHOLD)
    properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
        pcrown,
        ACAS_PATH,
        #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
        #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
        logfile = "./eval/acas_results_polycrownbern_final_mem_opt_combinedimp_smithboundsoverapproximate_opt_$THRESHOLD.jld2",
        max_properties = Inf,
        print_freq = 5,
        n_steps = STEPS,
        save_history = true,
        save_times = true,
        timeout = 300,
        loss_fun = bounds_loss,
    )
end
