
using NNPoly
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN, verify_vnnlib, PolyCROWNBern, bounds_loss_violation_stop

ACAS_PATH = "../vnncomp2022_benchmarks/benchmarks/acasxu"
loss_fun = bounds_loss_violation_stop
loss_fun_name = "bounds_loss_violation_stop"

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
pcrown = PolyCROWNBern(use_shortcut=use_shortcut)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
    logfile = "./eval/acas_results_polycrownbern_two_poly_layers.jld2",
    max_properties = 2,
    print_freq = 1,
    n_steps = 3,
    save_history = true,
    timeout = 300,
    loss_fun = loss_fun
)
#pcrown = PolyCROWN()
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    logfile = "./eval/acas_results_polycrown_optimised.jld2",
#    max_properties = 2,
#    print_freq = 1,
#    n_steps = 1,
#    save_history = true,
#    timeout = 300,
#    loss_fun = loss_fun
#)



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

println("---- PolyCROWN----")
#pcrown = PolyCROWN()
#properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
#    pcrown,
#    ACAS_PATH,
#    #logfile = "./eval/acas_results_polycrown_$loss_fun_name.jld2",
#    logfile = "./eval/acas_results_polycrown_optimised.jld2",
#    max_properties = Inf,
#    print_freq = 10,
#    n_steps = 50,
#    save_history = true,
#    timeout = 300,
#    loss_fun = loss_fun
#)
println("---- PolyCROWNBern ----")
pcrown = PolyCROWNBern(use_shortcut=use_shortcut)
properties, times, y_starts, ys, y_hists, t_hists = verify_vnnlib(
    pcrown,
    ACAS_PATH,
    #logfile = "./eval/acas_results_polycrownbern_slow_bounds_$loss_fun_name.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2",
    #logfile = "./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros_and_shortcut.jld2",
    logfile = "./eval/acas_results_polycrownbern_two_poly_layers.jld2",
    max_properties = Inf,
    print_freq = 10,
    n_steps = 50,
    save_history = true,
    timeout = 300,
    loss_fun = loss_fun
)
