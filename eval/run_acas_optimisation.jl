
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib, PolyCROWNBern, bounds_loss_violation_stop, violation_loss, bounds_loss

ACAS_PATH = "../vnncomp2022_benchmarks/benchmarks/acasxu"
loss_fun = bounds_loss_violation_stop
loss_fun_name = "bounds_loss_violation_stop"
POLY_LAYERS =2 

#=println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_tightened.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
solver = AlphaNeurify(use_tightened_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_no_tighten.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(dsolver, ACAS_PATH, logfile="./eval/acas_results_poly.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)

println("precompiling ...")
acrown = aCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(acrown, ACAS_PATH, logfile="./eval/acas_results_acrown.jld2", max_properties=2, print_freq=1, n_steps=10, save_history=true, timeout=300)
=#

use_shortcut = false
println("precompiling ...")
pcrown = PolyCROWNBern(use_dense_repr=true, use_combined_repr=false,use_shortcut=false, poly_layers = 1)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
    logfile = "./eval/acas_results_polycrownbern_one_layer_dense_repr_optimisation.jld2",
    max_properties = 1,
    print_freq = 1,
    n_steps = 3,
    save_history = true,
    save_times = true,
    timeout = 300,
    loss_fun = bounds_loss,
)
#pcrown = PolyCROWNBern(use_dense_repr=false, use_combined_repr=true,use_shortcut=false, poly_layers = 1)
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
#    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
#    logfile = "./eval/acas_results_polycrownbern_one_layer_full_dense_bounds.jld2",
#    max_properties = 1,
#    print_freq = 1,
#    n_steps = 3,
#    save_history = true,
#    save_times = true,
#    timeout = 300,
#    loss_fun = bounds_loss,
#)
#pcrown = PolyCROWNBern(use_shortcut=false, poly_layers = 2)
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
#    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
#    logfile = "./eval/acas_results_polycrownbern_one_layers.jld2",
#    max_properties = 1,
#    print_freq = 1,
#    n_steps = 3,
#    save_history = true,
#    timeout = 300,
#    loss_fun = bounds_loss,
#)
pcrown = PolyCROWN(DiffNNPolySym(common_generators=true), poly_layers=1)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    logfile = "./eval/acas_results_polycrown_one_layer_optimisation.jld2",
    max_properties = 2,
    print_freq = 1,
    n_steps = 1,
    save_history = true,
    save_times = true,
    timeout = 300,
    loss_fun = bounds_loss,
)



println("running experiments ...")

#=println("---- AlphaNeurify tightened ----")
solver = AlphaNeurify(use_tightened_bounds=true)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_tightened.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- AlphaNeurify no tighten ----")
solver = AlphaNeurify(use_tightened_bounds=false)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(solver, ACAS_PATH, logfile="./eval/acas_results_lin_no_tighten.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- DiffNNPolySym ----")
dsolver = DiffNNPolySym(truncation_terms=25, common_generators=true, save_bounds=false)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(dsolver, ACAS_PATH, logfile="./eval/acas_results_poly.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- α-CROWN ----")
acrown = aCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(acrown, ACAS_PATH, logfile="./eval/acas_results_acrown.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)

println("---- PolyCROWN ----")
pcrown = PolyCROWN()
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(pcrown, ACAS_PATH, logfile="./eval/acas_results_polycrown_own_run.jld2", max_properties=Inf, print_freq=50, n_steps=5000, save_history=true, timeout=300)
=#

#println("---- PolyCROWNBern ----")
#pcrown = PolyCROWNBern(use_shortcut=false,use_dense_repr=false,use_combined_repr=true, poly_layers = 1)
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
#    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
#    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros_and_shortcut.jld2",
#    logfile = "./eval/acas_results_polycrownbern_one_layer_full_dense_bounds.jld2",
#    max_properties = Inf,
#   print_freq = 10,
#    n_steps = 50,
#    save_history = true,
#    save_times = true,
#    timeout = 300,
#    loss_fun = bounds_loss,
#)
println("---- PolyCROWNBern dense_repr----")
pcrown = PolyCROWNBern(use_dense_repr=true, use_combined_repr=false,use_shortcut=false, poly_layers = 1)
#@profview properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros_and_shortcut.jld2",
    logfile = "./eval/acas_results_polycrownbern_one_layer_dense_repr_optimisation.jld2",
    max_properties = Inf,
    print_freq = 10,
    n_steps = 50,
    save_history = true,
    save_times = true,
    timeout = 300,
    loss_fun = bounds_loss,
)
#println("---- PolyCROWNBern ----")
#pcrown = PolyCROWNBern(use_shortcut=true, poly_layers = 3)
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
#    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
#    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros_and_shortcut.jld2",
#    logfile = "./eval/acas_results_polycrownbern_three_layers.jld2",
#    max_properties = Inf,
#    print_freq = 10,
#    n_steps = 50,
#    save_history = true,
#    timeout = 300,
#    loss_fun = bounds_loss,
#)

println("---- PolyCROWN----")
pcrown = PolyCROWN( DiffNNPolySym(common_generators=true), poly_layers=1)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrown_$loss_fun_name.jld2",
    logfile = "./eval/acas_results_polycrown_one_layer_optimisation.jld2",
    max_properties = Inf,
    print_freq = 10,
    n_steps = 50,
    save_history = true,
    save_times = true,
    timeout = 300,
    loss_fun = bounds_loss,
)
#pcrown = PolyCROWN( DiffNNPolySym(common_generators=true), poly_layers=2)
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    #logfile = "./eval/acas_results_polycrown_$loss_fun_name.jld2",
#    logfile = "./eval/acas_results_polycrown_two_layers_act.jld2",
#    max_properties = Inf,
#    print_freq = 10,
#    n_steps = 50,
#    save_history = true,
#    timeout = 300,
#    loss_fun = bounds_loss,
#)
#
