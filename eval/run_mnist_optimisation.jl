
using NNPoly, Profile
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN,PolyCROWNBern, verify_vnnlib

MNIST_PATH = "./eval/mnist_fc"

#println("precompiling ...")
#solver = PolyCROWN()
#properties, times, y_starts, ys, y_hists = verify_vnnlib(
#    solver,
#    MNIST_PATH,
#    logfile = "./eval/mnist_results_PolyCROWN_jul_28.jld2",
#    max_properties = 1,
#    print_freq = 1,
#    n_steps = 10,
#    save_history = true,
#    timeout = 300,
#    only_pattern = "256x6",
#    force_gc = true,
#)


solver = PolyCROWNBern(use_shortcut=false,poly_layers=1,use_dense_repr=false, use_combined_repr=true)
#@profview properties, times, y_starts, ys, y_hists = verify_vnnlib(
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist_results_PolyCROWNBern_faster_dense_bounds_jul_30-p.jld2",
    #logfile = "./eval/mnist256x6_results_PolyCROW.jld2",
    max_properties = 1,
    print_freq = 1,
    n_steps = 1,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)
solver = PolyCROWNBern(use_shortcut=false,poly_layers=1,use_dense_repr=false, use_combined_repr=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist_results_PolyCROWNBern_faster_dense_bounds_jul_30-p.jld2",
    #logfile = "./eval/mnist256x6_results_PolyCROW.jld2",
    max_properties = 1,
    print_freq = 1,
    n_steps = 1,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)
exit()




println("running experiments ...")

#println("PolyCROWN ...")
#solver = PolyCROWN()
#properties, times, y_starts, ys, y_hists = verify_vnnlib(
#    solver,
#    MNIST_PATH,
#    logfile = "./eval/mnist_results_PolyCROWN_jul_24.jld2",
#    max_properties = Inf,
#    print_freq = 5,
#    n_steps = 1000,
#    save_history = true,
#    timeout = 300,
#    only_pattern = "256x6",
#    force_gc = true,
#)

println("PolyCROWNBern ...")
solver = PolyCROWNBern(use_shortcut=false,poly_layers=1,use_dense_repr=false,use_combined_repr=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist_results_PolyCROWNBern_faster_dense_bounds_jul_30.jld2",
    max_properties = Inf,
    print_freq = 5,
    n_steps = 1000,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)
exit()
solver = PolyCROWNBern(use_shortcut=false,poly_layers=1,use_dense_repr=true,use_combined_repr=false)
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist_results_PolyCROWNBern_dense_repr.jld2",
    max_properties = Inf,
    print_freq = 5,
    n_steps = 1000,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)
