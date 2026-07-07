using JLD2, Plots, DataFrames, CSV
ENV["GKSwstype"] = "100" # prevent gui from opening
polycrown_res  = load("./eval/acas_results_polycrown_performance_run.jld2") 
polycrown_bern_res  = load("./eval/acas_results_polycrownbern_performance_run.jld2") 

plot(polycrown_res["ys"],polycrown_bern_res["ys"], seriestype=:scatter, title="acas Summed bound widths")
xl = xlims()
yl = ylims()

lo = min(xl[1], yl[1])
hi = max(xl[2], yl[2])
plot!([lo,hi],[lo,hi])
xlabel!("Polycrown widths")
ylabel!("Bern widths")

savefig("plots/acas_results.svg")



polycrown_violation_loss  = load("./eval/acas_results_polycrown_violation_loss.jld2") 
polycrown_bern_violation_loss  = load("./eval/acas_results_polycrownbern_performance_run.jld2") 

plot(polycrown_violation_loss["ys"],polycrown_bern_violation_loss["ys"], seriestype=:scatter, title="acas violation loss")
xl = xlims()
yl = ylims()

lo = min(xl[1], yl[1])
hi = max(xl[2], yl[2])
plot!([lo,hi],[lo,hi])
xlabel!("Polycrown")
ylabel!("Bern")

savefig("plots/acas_violation_loss.svg")


polycrown_bounds_loss_violation_stop  = load("./eval/acas_results_polycrown_bounds_loss_violation_stop.jld2") 
polycrown_bern_bounds_loss_violation_stop  = load("./eval/acas_results_polycrownbern_bounds_loss_violation_stop.jld2") 
print(minimum(polycrown_bounds_loss_violation_stop["ys"]))

eps = 1e-3 
# replace zeros for plotting 
xp = max.(polycrown_bounds_loss_violation_stop["ys"], eps/2)
yp = max.(polycrown_bern_bounds_loss_violation_stop["ys"], eps/2)

xticks = (
    [eps/2, 1e-2, 1e-1, 1, 10, 100, 1000,1e+4,1e+5,1e+6],
    ["0", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³", "10⁴","10⁵","10⁶"]
)

yticks = (
    [eps/2, 1e-2, 1e-1, 1, 10, 100, 1000,1e+4,1e+5,1e+6],
    ["0", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³",  "10⁴","10⁵","10⁶"]
)
plot(xp,yp,xticks=xticks,aspect_ratio=:equal, yticks=yticks, seriestype=:scatter, title="acas bounds loss violation stop")
plot!(xscale=:log10,yscale=:log10, legend=false)
xlims!(eps/4, 2e6)
ylims!(eps/4, 2e6)
xl = xlims()
yl = ylims()

vspan!( [eps/10, eps];
       color=:sandybrown,
       alpha=0.35)

# bottom horizontal strip (y≈0)
hspan!( [eps/10, eps];
       color=:sandybrown,
       alpha=0.35)

lo = min(xl[1], yl[1])
hi = max(xl[2], yl[2])
plot!([lo,hi],[lo,hi])
xlabel!("Polycrown")
ylabel!("Bern")

savefig("plots/acas_bounds_loss_violation_stop.svg")

function plot_bound_widths(xs,ys,x_label,y_label,title,file_name)

    eps = 1e-3 
    # replace zeros for plotting 
    xp = max.(xs, eps/2)
    yp = max.(ys, eps/2)

    xticks = (
	[eps/2, 1e-2, 1e-1, 1, 10, 100, 1000,1e+4,1e+5,1e+6],
	["0", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³", "10⁴","10⁵","10⁶"]
    )

    yticks = (
	[eps/2, 1e-2, 1e-1, 1, 10, 100, 1000,1e+4,1e+5,1e+6],
	["0", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³",  "10⁴","10⁵","10⁶"]
    )
    plot(xp,yp,xticks=xticks, aspect_ratio=:equal, yticks=yticks, seriestype=:scatter, title=title)
    plot!(xscale=:log10,yscale=:log10, legend=false)
    xlims!(eps/4, 2e6)
    ylims!(eps/4, 2e6)
    xl = xlims()
    yl = ylims()

    vspan!( [eps/10, eps];
	   color=:sandybrown,
	   alpha=0.35)

    # bottom horizontal strip (y≈0)
    hspan!( [eps/10, eps];
	   color=:sandybrown,
	   alpha=0.35)

    lo = min(xl[1], yl[1])
    hi = max(xl[2], yl[2])
    plot!([lo,hi],[lo,hi])
    xlabel!(x_label)
    ylabel!(y_label)

    savefig(file_name)
end

plot_bound_widths(
    load("./eval/acas_results_polycrownbern_slow_bounds_bounds_loss_violation_stop.jld2")["ys"], 
    load("./eval/acas_results_polycrownbern_bounds_loss_violation_stop.jld2")["ys"],
    "Bern slow bounds", "Bern faster bounds", 
    "acas slower bern bounds",
    "plots/acas_bern_slow_vs_fast.svg"  )
plot_bound_widths(
    load("./eval/acas_results_polycrownbern_slow_bounds_bounds_loss_violation_stop.jld2")["ys"], 
    load("./eval/acas_results_polycrown_bounds_loss_violation_stop.jld2")["ys"],
    "Bern slow bounds", "Polycorwn", 
    "acas slower bern bounds",
    "plots/acas_bern_slow_vs_poly_crown.svg"  )

#plot_bound_widths(
#    load("./eval/mnist256x2_15_results_PolyCROWN.jld2")["ys"],
#    load("./eval/mnist256x2_15_results_PolyCROWNBern.jld2")["ys"], 
#    "polycrown", "bern", 
#    "mnist256x2 15 bern vs poly",
#    "plots/mnist256x2_bern_vs_poly.svg"  )

    pcrown = load("./eval/mnist256x6_results_PolyCROWN.jld2")
    pcrown_bern = load("./eval/mnist256x6_results_PolyCROWNBern.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern["ys"],
    "polycrown", "bern", 
    "mnist 256x6 15,20 bern vs poly",
    "plots/mnist256x6_bern_vs_poly.svg"  )
#plot_bound_widths(
#    #load("./eval/mnist256x6_results_PolyCROWN.jld2")["ys"],
#    load("eval/mnist256x6_results_PolyCROWNBern_more_combine.jld2")["ys"],
#    load("./eval/mnist256x6_results_PolyCROWNBern.jld2")["ys"], 
#    "bern more combine", "bern", 
#    "mnist bern more combine vs bern",
#    "plots/mnist_more_combine_bern_vs_bern.svg"  )
#


polycrown_res  = load("./eval/mnist256x6_results_PolyCROWN.jld2")["ys"]
polycrown_bern_res  = load("./eval/mnist256x6_results_PolyCROWNBern.jld2")["ys"]

plot(polycrown_res,polycrown_bern_res, seriestype=:scatter, title="mnist bound widths")
xl = xlims()
yl = ylims()

lo = min(xl[1], yl[1])
hi = max(xl[2], yl[2])
plot!([lo,hi],[lo,hi])
xlabel!("Polycrown widths")
ylabel!("Bern widths")

savefig("plots/mnist_poly_vs_bern_linear_scale.svg")


    pcrown_bern = load("./eval/mnist256x6_results_PolyCROWNBern.jld2")
    pcrown_bern_combined = load("./eval/mnist256x6_results_PolyCROWNBern_combined.jld2")
plot_bound_widths(
    pcrown_bern["ys"],
    pcrown_bern_combined["ys"],
    "bern original", "bern combined", 
    "mnist 256x6 15,20 bern original vs bern combined ",
    "plots/mnist256x6_bern_vs_bern_combined.svg"  )

pcrown = load("./eval/mnist256x6_results_PolyCROWN.jld2")
pcrown_bern_combined = load("./eval/mnist256x6_results_PolyCROWNBern_combined.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern_combined["ys"],
    "polycrown", "bern combined", 
    "mnist 256x6 15,20 polycrown vs bern combined ",
    "plots/mnist256x6_poly_vs_bern_combined.svg"  )

pcrown = load("./eval/acas_results_polycrown_bounds_loss_violation_stop.jld2")
pcrown_bern_combined = load("./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern_combined["ys"],
    "polycrown", "bern combined + combine_terms no shortcut", 
    "acas polycrown vs bern combined + combine_terms no shortcut",
    "plots/acas_polycrown_vs_bern_combined_intervals_with_combine_terms.svg"  )

println("bern with B per Low and Up + shortcut & acas & $(sum(pcrown_bern_combined["times"]) / length(pcrown_bern_combined["times"]) )")

pcrown = load("./eval/acas_results_polycrown_bounds_loss_violation_stop.jld2")
pcrown_bern_combined = load("./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_shortcut.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern_combined["ys"],
    "polycrown", "bern combined + combine_terms + shortcut", 
    "acas polycrown vs bern combined + combine_terms + shortcut",
    "plots/acas_polycrown_vs_bern_combined_intervals_with_combine_terms_and_shortcut.svg"  )

pcrown = load("./eval/mnist256x6_results_PolyCROWN_all.jld2")
pcrown_bern_combined = load("./eval/mnist256x6_results_PolyCROWNBern_combined_all.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern_combined["ys"],
    "polycrown", "bern combined", 
    "mnist256x6 10,15,20,25 polycrown vs bern combined",
    "plots/mnist256x6_all_polycrown_vs_bern_combined.svg"  )

println("polycron & mnist256x6 & $(sum(pcrown["times"]) / length(pcrown["times"]) )")
println("bern with B per Low and Up & mnist256x6 & $(sum(pcrown_bern_combined["times"]) / length(pcrown_bern_combined["times"]) )")

pcrown = load("./eval/mnist256x6_results_PolyCROWN_all.jld2")
pcrown_bern_combined = load("./eval/mnist256x6_results_PolyCROWNBern_new_repr.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern_combined["ys"],
    "polycrown", "bern with shared B between Low and Up", 
    "mnist256x6 10,15,20,25 polycrown vs bern with one B",
    "plots/mnist256x6_all_polycrown_vs_bern_new_repr.svg"  )

println("bern with shared B between Low and Up + shortcut + remove_zero terms& mnist256x6 & $(sum(pcrown_bern_combined["times"]) / length(pcrown_bern_combined["times"]) )")

pcrown = load("./eval/acas_results_polycrown_violation_loss.jld2")
pcrown_bern_combined = load("./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern_combined["ys"],
    "polycrown", "bern with shared B between Low and Up", 
    "acas vs bern with shared B between Low and Up",
    "plots/acas_polycrown_vs_bern_with_shared_B.svg"  )

println("polycron & acas & $(sum(pcrown["times"]) / length(pcrown["times"]) )")
println("bern with shared B between Low and Up  & acas & $(sum(pcrown_bern_combined["times"]) / length(pcrown_bern_combined["times"]) )")



pcrown_bern_no_shortcut = load("./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros.jld2")
pcrown_bern_combined = load("./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros_and_shortcut.jld2")
plot_bound_widths(
    pcrown_bern_no_shortcut["ys"],
    pcrown_bern_combined["ys"],
    "bern with shared B  ", "bern with shared B + shortcut", 
    "acas bern with shared B vs bern with shared B + shortcut",
    "plots/acas_bern_with_shared_B_and_shortcut_vs_bern_with_shared_B.svg"  )

println("bern with shared B between Low and Up + shortcut & acas & $(sum(pcrown_bern_combined["times"]) / length(pcrown_bern_combined["times"]) )")



pcrown =  CSV.read("./eval/mnist_fc_growing/logs_2026-07-01_13-58-40_with_hist.csv", DataFrame)
pcrown_bern =  CSV.read("./logs_2026-07-06_15-15-29_with_hist.csv", DataFrame)
 

eps = 1e-3 
# replace zeros for plotting 
xp = max.(pcrown.y, eps/2)
yp = max.(pcrown_bern.y, eps/2)

xticks = (
    [eps/2, 1e-2, 1e-1, 1, 10, 100, 1000,1e+4,1e+5,1e+6],
    ["0", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³", "10⁴","10⁵","10⁶"]
)

yticks = (
    [eps/2, 1e-2, 1e-1, 1, 10, 100, 1000,1e+4,1e+5,1e+6],
    ["0", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³",  "10⁴","10⁵","10⁶"]
)
plot(xp,yp,xticks=xticks, aspect_ratio=:equal, yticks=yticks, marker_z=pcrown.n_unfixed,seriestype=:scatter, title="mnist growing")
plot!(xscale=:log10,yscale=:log10, legend=false)
xlims!(eps/4, 2e6)
ylims!(eps/4, 2e6)
xl = xlims()
yl = ylims()

vspan!( [eps/10, eps];
       color=:sandybrown,
       alpha=0.35)

# bottom horizontal strip (y≈0)
hspan!( [eps/10, eps];
       color=:sandybrown,
       alpha=0.35)

lo = min(xl[1], yl[1])
hi = max(xl[2], yl[2])
plot!([lo,hi],[lo,hi])
xlabel!("polycrown")
ylabel!("bern")

savefig("plots/mnist_growing.svg")
println("polycrown & mnist_growing & $(sum(pcrown.time) / length(pcrown.time) )")
println("bern with shared B + shortcut & mnist_growing & $(sum(pcrown_bern.time) / length(pcrown_bern.time) )")

pcrown_bern = load("./eval/acas_results_polycrownbern_combined_intervals_with_combine_terms_and_remove_zeros_and_shortcut.jld2")
pcrown_bern_optimised = load("./eval/acas_results_polycrownbern_combined_intervals_optimisation.jld2")
plot_bound_widths(
    pcrown_bern["ys"],
    pcrown_bern_optimised["ys"],
    "bern with shared B + shortcut", "bern with shared B + shortcut optimisition", 
    "acas bern with shared B + shortcut vs optimised",
    "plots/acas_optimised.svg"  )

println("bern with shared B  + shortcut  optimised & acas & $(sum(pcrown_bern_optimised["times"])/50 / length(pcrown_bern_optimised["times"]) )")



pcrown = load("./eval/acas_results_polycrown_violation_loss.jld2")
pcrown_bern = load("./eval/acas_results_polycrownbern_two_poly_layers.jld2")
plot_bound_widths(
    pcrown["ys"],
    pcrown_bern["ys"],
    "polycrown", "bern two poly layers", 
    "acas pcrown vs bern two poly layers",
    "plots/acas_polycrown_vs_bern_two_poly_layers.svg"  )

