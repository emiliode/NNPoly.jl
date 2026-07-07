
using CSV
using DataFrames
using JLD2

if length(ARGS) != 1
    println("Usage: julia add_hist_columns.jl <results.csv>")
    exit(1)
end

csv_file = ARGS[1]

df = CSV.read(csv_file, DataFrame)

# Allocate new columns
df.time_file = Vector{Union{Missing, Float64}}(missing, nrow(df))
df.y = Vector{Union{Missing, Float64}}(missing, nrow(df))

for i in 1:nrow(df)
    hist_path = df.hist_file[i]

    if !ismissing(hist_path) && isfile(hist_path)
        data = load(hist_path)

        if haskey(data, "t_hist") && !isempty(data["t_hist"])
            df.time_file[i] = data["t_hist"][1]   # or last(data["t_hist"])
        end

        if haskey(data, "y_hist") && !isempty(data["y_hist"])
            df.y[i] = data["y_hist"][1]           # or last(data["y_hist"])
        end
    else
        @warn "Could not find history file: $hist_path"
    end
end

# Write to "<original>_with_hist.csv"
base, ext = splitext(csv_file)
output_file = base * "_with_hist" * ext

CSV.write(output_file, df)

println("Wrote updated CSV to: $output_file")
