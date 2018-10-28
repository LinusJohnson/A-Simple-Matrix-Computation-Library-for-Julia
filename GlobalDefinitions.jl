module GlobalDefinitions
export @def, @StandardIterationSettingsFields, MatrixProblem,
        getMethodClass, LinAlgMethod, IterationSettings,
        DefaultIterationSettings, IterationInfo,
        setStandardIterationSettingsFields!, ErrFun, IterationState,
        struct2dict, EigMethod, KrylovProjectionMethod, EigValSettings,
        KrylovProjectionSettings, run, TransformationMethod,
        TransformationSettings, MatrixFunctionMethod, MatrixFunctionSettings, pm
using Parameters

macro def(name, definition)
    quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

abstract type MatrixProblem end
abstract type LinAlgMethod end
abstract type ErrFun end
abstract type IterationSettings end
abstract type EigMethod <: LinAlgMethod end
abstract type KrylovProjectionMethod <: LinAlgMethod end
abstract type TransformationMethod <: LinAlgMethod end
abstract type MatrixFunctionMethod <: LinAlgMethod end
abstract type EigValSettings <: IterationSettings end
abstract type KrylovProjectionSettings <: IterationSettings end
abstract type TransformationSettings <: IterationSettings end
abstract type MatrixFunctionSettings <: IterationSettings end
abstract type IterationState end

@def StandardIterationSettingsFields begin
    maxIter::Nullable{Integer}
    maxErr::Nullable{AbstractFloat}
    method::LinAlgMethod
    errFun::Nullable{ErrFun}
    storeIterates::Bool
    time::Bool
    verbose::Int16
end

@with_kw type DefaultIterationSettings <: IterationSettings
    @StandardIterationSettingsFields
end

type IterationInfo
    k::Vector{Integer}
    iterations::Vector{Any}
    errors::Vector{Float64}
    t::Vector{Float64}
end
IterationInfo() =
IterationInfo(Integer[],IterationState[],Float64[],Float64[])

function setStandardIterationSettingsFields!(
            params::Dict{Symbol,Any},
            defaultSettings::DefaultIterationSettings,
            maxIter::Union{Integer,Void},
            maxErr::Union{AbstractFloat,Void},
            method::Union{LinAlgMethod,Void},
            errFun::Union{ErrFun,Void},
            storeIterates::Union{Bool,Void},
            time::Union{Bool,Void},
            verbose::Union{Bool,Void})
    if method == nothing
        params[:method] = defaultSettings.method
    else params[:method] = method end
    if errFun == nothing
        params[:errFun] = defaultSettings.errFun
    else params[:errFun] = errFun end
    if storeIterates == nothing
        params[:storeIterates] = defaultSettings.storeIterates
    else params[:storeIterates] = storeIterates end
    if time == nothing
        params[:time] = defaultSettings.time
    else params[:time] = time end
    if verbose == nothing
        params[:verbose] = defaultSettings.verbose
    else params[:verbose] = verbose end
    # set given or default stopping critera
    if maxIter == nothing && !isnull(defaultSettings.maxIter)
        params[:maxIter] = get(defaultSettings.maxIter)
    else params[:maxIter] = maxIter end
    if maxErr == nothing && !isnull(defaultSettings.maxErr)
        params[:maxErr] = get(defaultSettings.maxErr)
    else params[:maxErr] = maxErr end

    # if one stopping critera is given use only that one
    if maxIter != nothing && maxErr == nothing
        params[:maxErr] = Nullable{AbstractFloat}()
    elseif maxIter == nothing && maxErr != nothing
        params[:maxIter] = Nullable{Integer}()
    elseif maxIter == nothing && maxErr == nothing
        params[:maxErr] = Nullable{AbstractFloat}()
    end

    #println(stacktrace())
    #println(maxIter)
    #println(maxErr)
    #println(params[:maxIter])
end

function allSubtypes(type_::DataType)
    subtypes_ = subtypes(type_)
    for subtype = subtypes(type_)
        subtypes_ = append!(subtypes_, allSubtypes(subtype))
    end
    return subtypes_
end

function getMethodClass(method::LinAlgMethod, methodClasses::Vector{DataType})
    for methodClass = methodClasses
        if typeof(method) in allSubtypes(methodClass)
            return methodClass
        elseif typeof(method) == methodClass
            return typeof(method)
        end
    end
    return supertype(methodClasses[1])
end

function pm(M)
    M_p = deepcopy(M); M_p[abs.(M_p) .< eps()*1e1] = 0
    println(); show(IOContext(STDOUT, limit=true), "text/plain", M_p); println()
end

errorFunction(state::IterationState) = errorFunction(state::IterationState, get(state.settings.errFun))
errorFunction(::IterationState, ::ErrFun) = error("Not implemented")
getResult(::IterationState) = error("Not implemented")
step!(::IterationState, ::Integer) = error("Not implemented")

@inline function next(state::IterationState, k::Integer)
    if state.settings.time
        t_ = @elapsed step!(state, k)
        push!(state.info.t, t_ + (k==1 ? 0 : state.info.t[k-1]))
    else
        step!(state, k)
    end
    push!(state.info.k, k)
    if state.settings.storeIterates
        push!(state.info.iterations, getResult(state)) end
    if !isnull(state.settings.errFun)
        push!(state.info.errors, errorFunction(state)) end
        if state.settings.verbose >= 3
            println("Iteration: $(k),   Error: $(state.info.errors[end]),   Target: $(get(state.settings.maxErr))")
        end
    return nothing, k+1
end

start(::IterationState) = 1

@inline function done(state::IterationState, k::Integer)
    if isnull(state.settings.maxErr) && isnull(state.settings.maxIter)
        error("Stopping condition not set")
    end
    res = true
    if !isnull(state.settings.maxErr) && !isempty(state.info.errors)
        res &= state.info.errors[end] <= get(state.settings.maxErr)
    end
    if !isnull(state.settings.maxIter)
        res &= k > get(state.settings.maxIter)
    end
    return res & (k != 1)
end

@inline function run(prob::MatrixProblem)
    iter = IterationState(prob)
    #for i = iter end
    state = start(iter)
    while !done(iter, state)
        (_, state) = next(iter, state)
    end
    return getResult(iter), iter.info
end

function struct2dict(strct::Any)
    fields = fieldnames(strct)
    vals = [getfield(strct,field) for field = fields]
    return Dict(zip(fields,vals))
end

end
