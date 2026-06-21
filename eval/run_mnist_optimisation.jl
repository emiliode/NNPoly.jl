
using NNPoly, Profile
import NNPoly: DiffNNPolySym, AlphaNeurify, aCROWN, PolyCROWN,PolyCROWNBern, verify_vnnlib

MNIST_PATH = "./eval/mnist_fc"

println("precompiling ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist256x6_results_PolyCROWN.jld2",
    max_properties = 2,
    print_freq = 1,
    n_steps = 10,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)

solver = PolyCROWNBern(use_shortcut=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist256x6_results_PolyCROWNBern.jld2",
    #logfile = "./eval/mnist256x6_results_PolyCROW.jld2",
    max_properties = 1,
    print_freq = 1,
    n_steps = 1,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)




println("running experiments ...")
#
println("PolyCROWN ...")
solver = PolyCROWN()
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist256x6_results_PolyCROWN.jld2",
    max_properties = Inf,
    print_freq = 5,
    n_steps = 1000,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)

println("PolyCROWNBern ...")
solver = PolyCROWNBern(use_shortcut=true)
properties, times, y_starts, ys, y_hists = verify_vnnlib(
    solver,
    MNIST_PATH,
    logfile = "./eval/mnist256x6_results_PolyCROWNBern.jld2",
    #logfile = "./eval/mnist256x6_results_PolyCROWN.jld2",
    max_properties = Inf,
    print_freq = 5,
    n_steps = 1000,
    save_history = true,
    timeout = 300,
    only_pattern = "256x6",
    force_gc = true,
)
