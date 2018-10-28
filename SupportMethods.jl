module SupportMethods
using ..EigenvalueMethods
#using Gadfly
using DataFrames
export save2file, errorPlot, DataFrame2LatexHW2Problem2,
       DataFrame2LatexHW1Problem3, DataFrame2LatexHW3Problem3c,
       DataFrame2LatexHW3Problem3d

function save2file(savefunc, obj, subfolder::String, file_name::String, top_target_dir::String)
    current_dir = pwd()
    base_dir = ""
    if basename(current_dir) != top_target_dir
        for (root, dirs, files) in walkdir(current_dir)
            for dir in dirs
                if basename(dir) == top_target_dir
                    base_dir = joinpath(root, dir); break
                end
            end
            if base_dir != "" break end
        end
    end
    path = joinpath(base_dir, subfolder)
    if !isdir(path) mkdir(path) end
    cd(()->savefunc(file_name, obj), path)
    return nothing
end

#function errorPlot(prob::EigValProblem, file_name::String, top_target_dir::String)
#    res, info = getEigApprox(prob)
#    df = DataFrame(k=info.iterations, error=info.errors)
#    th = Theme(highlight_width = 0pt, major_label_font_size = 21pt,
#               minor_label_font_size = 19pt)
#    p = plot(df, x=:k, y=:error, th, Geom.point, Scale.y_log10)
#    save2file((file_name,obj)->draw(PDF(file_name, 350mm, (350/2)mm), obj), p, "Report/Figures", "$file_name.pdf", top_target_dir)
#end

function DataFrame2LatexHW1Problem3(df::DataFrame, name::String)
    min_orth = minimum(df[:orth])
    exp10 = 10. .^(-(1:40));
    expn = exp10[indmin(abs(min_orth-exp10))]
    methods = unique(df[:method])
    orderdict = Dict(x => i for (i,x) in enumerate(methods))
    s(df_) = sort(df_; cols = [:method], by = x->orderdict[x])
    n_cols = 1+2*length(methods)
    col_ind_string = repeat(" l ", n_cols)
    latex = "\\sisetup{round-mode = figures, round-precision = 3}\n"
    latex *= "\\begin{tabular}{@{}"*col_ind_string*"@{}}\n"
    col_1_string = join(["\\multicolumn{2}{l}{$method} &" for method in methods])[1:end-1]
    col_2_string = repeat("& time/\\si{\\second} & orth \$\\times\$ \$ \\num[retain-unity-mantissa = false]{"*"$("$expn"[4:end])"*"} \$ ", length(methods))
    formatter(num_, col) = col == :orth ? "\\num{$(num_/expn)}" : "\\num{$(num_)}" # let latex take care of formatting
    rows = ""
    for mdf in groupby(df,:m)
        rows *= "\$m = $(mdf[1,:m])\$ & "
        for method in methods
            mmethoddf = mdf[mdf[:method] .== method, :]
            rows *= join(["$(formatter(mmethoddf[1,col],col)) & " for col in [:time, :orth]])
        end
        rows = rows[1:end-2] * "\\\\ \n"
    end
    #rows = join(["\$m = $(mdf[1,:m])\$ & " * join([join(["$(formatter(mmethoddf[1,col],col)) & " for col in [:time, :orth]])
    #            for mmethoddf in groupby(s(mdf),:method)])[1:end-2] * "\\\\ \n"
    #                    for mdf in groupby(df,:m)])
    latex *= "\\toprule\n"
    latex *= "\\multicolumn{1}{l}{} & " * col_1_string * " \\\\\n"
    latex *= "\\midrule\n"; latex *= "\\midrule\n"
    latex *= col_2_string * " \\\\\n"
    latex *= "\\midrule\n"
    latex *= rows
    latex *= "\\bottomrule\n"
    latex *= "\\end{tabular}\n"

    save2file((file_name,obj)->write(file_name,obj), latex, "Report", name*".tex", "Homework 1")
end

function DataFrame2LatexHW2Problem2(df::DataFrame, name::String)
    m_vals = unique(df[:m])
    n_cols = 1+2*length(m_vals)
    col_ind_string = repeat(" l ", n_cols)
    latex = "\\sisetup{round-mode = figures, round-precision = 3}\n"
    latex *= "\\begin{tabular}{@{}"*col_ind_string*"@{}}\n"
    formatter(num_) = "\\num{$num_}" #@sprintf("%.3f",num_) let latex take care of formatting
    for αdf in groupby(df,[:α])
        col_1_string = join(["\\multicolumn{2}{l}{m=$m} &" for m in m_vals])[1:end-1]
        col_2_string = repeat("& resnorm & time/\\si{\\milli\\second} ", length(m_vals))
        if :n in names(αdf)
            rows = join(["\$n = $(nαdf[1,:n])\$ & " * join([join(["$(formatter(mnαdf[1,col])) & " for col in [:resnorm, :time]])
                        for mnαdf in groupby(nαdf,:m)])[1:end-2] * "\\\\ \n"
                                for nαdf in groupby(αdf,:n)])
        else
            rows = "& " * join([join(["$(formatter(mαdf[1,col])) & " for col in [:resnorm, :time]])
                        for mαdf in groupby(αdf,:m)])[1:end-2] * "\\\\ \n"
        end
        latex *= "\\toprule\n"
        latex *= "\\multicolumn{$(n_cols)}{c}{\$\\alpha = $(αdf[1,:α])\$} \\\\\n"
        latex *= "\\midrule\n"
        latex *= "\\multicolumn{1}{l}{} & "*col_1_string*" \\\\\n"
        latex *= "\\midrule\n"; latex *= "\\midrule\n"
        latex *= col_2_string * " \\\\\n"
        latex *= "\\midrule\n"
        latex *= rows
        latex *= "\\bottomrule\n"
    end
    latex *= "\\end{tabular}\n"

    save2file((file_name,obj)->write(file_name,obj), latex, "Report", name*".tex", "Homework 2")
end

function DataFrame2LatexHW3Problem3c(df::DataFrame, name::String)
    algos = unique(df[:algo])
    n_cols = 1+length(algos)
    col_ind_string = repeat(" l ", n_cols)
    latex = "\\sisetup{round-mode = figures, round-precision = 3}\n"
    latex *= "\\begin{tabular}{@{}"*col_ind_string*"@{}}\n"
    f(num_) = "\\num{$num_}" #let latex take care of formatting
    col_string = "& " * join(["CPU-time for $algo/\\si{\\second} &" for algo in algos])[1:end-1]
    rows = ""
    for mdf in groupby(df,:m)
        rows *= "\$m = $(mdf[1,:m])\$ & "
        for algo in algos
            algomdf = mdf[mdf[:algo] .== algo, :]
            rows *= "$(f(algomdf[1,:time])) & "
        end
        rows = rows[1:end-2] * "\\\\ \n"
    end

    latex *= "\\toprule\n"
    latex *= col_string * " \\\\\n"
    latex *= "\\midrule\n"
    latex *= "\\midrule\n"
    latex *= rows
    latex *= "\\bottomrule\n"
    latex *= "\\end{tabular}\n"
    save2file((file_name,obj)->write(file_name,obj), latex, "Report", name*".tex", "Homework 3")
end

function DataFrame2LatexHW3Problem3d(df::DataFrame, name::String)
    Σ = unique(df[:σ])
    Ε = unique(df[:ϵ])
    n_cols = length(Σ)
    col_ind_string = " l || "*repeat(" l ", n_cols)
    latex = "\\sisetup{round-mode = figures, round-precision = 3}\n"
    latex *= "\\begin{tabular}{@{}"*col_ind_string*"@{}}\n"
    f(num_) = "\\num{$num_}" #let latex take care of formatting
    col_1_string = "\\multicolumn{1}{l||}{\$\\epsilon\$} & \\multicolumn{2}{c}{\$|\\bar{h}_{2,1}|\$}"
    col_2_string = "& \$\\sigma = 0\$ & \$\\sigma = a_{2,2}\$"
    rows = ""
    for ϵ in Ε
        ϵdf = df[df[:ϵ] .== ϵ, :]
        rows *= "$(f(ϵdf[1,:ϵ])) & "
        for σ ∈ Σ
            σϵdf = ϵdf[ϵdf[:σ] .== σ, :]
            rows *= "$(f(σϵdf[1,:h₂₁])) & "
        end
        rows = rows[1:end-2] * "\\\\ \n"
    end
    latex *= col_1_string * " \\\\\n"
    latex *= "\\midrule\n"
    latex *= col_2_string * " \\\\\n"
    latex *= "\\midrule\n"
    latex *= "\\midrule\n"
    latex *= rows
    latex *= "\\end{tabular}\n"
    save2file((file_name,obj)->write(file_name,obj), latex, "Report", name*".tex", "Homework 3")
end

end
