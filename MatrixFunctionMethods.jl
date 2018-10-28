module MatrixFunctionMethods
export schurParlett, MatrixFunctionProblem, HessenbergReductionState,
       MatrixFunctionSettings, MatrixFunctionState
using ..OrthogonalizationMethods
using ..GlobalDefinitions
const GD = GlobalDefinitions
using Parameters
using Compat
import Base: start, next, done, llvmcall

type SchurParlett <: MatrixFunctionMethod end

const METHOD_CLASSES = [SchurParlett]
getMethodClass(method::MatrixFunctionMethod) = GD.getMethodClass(method, METHOD_CLASSES)
const DEFAULT_NUM_STEPS = nothing
const DEFAULT_ERR = nothing
const DEFAULT_METHOD = SchurParlett()
const DEFAULT_ERR_FUN = nothing
const DEFAULT_STORE_ITERATES = false
const DEFAULT_TIME = false
const DEFAULT_VERBOSE = 0
const DEFAULT_ITER_SETTINGS =
DefaultIterationSettings(maxIter=Nullable{Integer}(DEFAULT_NUM_STEPS),
                         maxErr=Nullable{AbstractFloat}(DEFAULT_ERR),
                         method=DEFAULT_METHOD,
                         errFun=Nullable{ErrFun}(DEFAULT_ERR_FUN),
                         storeIterates=DEFAULT_STORE_ITERATES,
                         time=DEFAULT_TIME,
                         verbose=DEFAULT_VERBOSE)

@with_kw type SchurParlettSettings <: MatrixFunctionSettings
    @StandardIterationSettingsFields
end

const settingsMap = Dict(SchurParlett => SchurParlettSettings)

function MatrixFunctionSettings(;
        maxIter::Union{Integer,Void}=nothing, maxErr::Union{AbstractFloat,Void}=nothing,
        method::Union{MatrixFunctionMethod,Void}=nothing, errFun::Union{ErrFun,Void}=nothing,
        storeIterates::Union{Bool,Void}=nothing, time::Union{Bool,Void}=nothing,
        verbose::Union{Int16,Void}=nothing)
    params = Dict{Symbol,Any}([])
    setStandardIterationSettingsFields!(params, DEFAULT_ITER_SETTINGS, maxIter,
                                        maxErr, method, errFun, storeIterates, time, verbose)
    settingsType = settingsMap[getMethodClass(params[:method])]
    return settingsType(;params...)
end

struct MatrixFunctionProblem <: MatrixProblem
    T::Type
    A::AbstractMatrix{<:Number}
    f::Function
    settings::MatrixFunctionSettings
end
setIter!(A::AbstractMatrix, settings::MatrixFunctionSettings) =
setIter!(A, settings, settings.method)
function setIter!(A::AbstractMatrix, settings::MatrixFunctionSettings, ::SchurParlett)
    settings.maxIter = size(A,1) - 1
    settings.maxErr = nothing
end

function MatrixFunctionProblem{T<:Number}(A::AbstractMatrix{T}, f::Function; kwargs...)
    settings = MatrixFunctionSettings(;kwargs...)
    setIter!(A,settings)
    return MatrixFunctionProblem(T, A, f, settings)
end
function MatrixFunctionProblem(A::AbstractMatrix{<:Number}, f::Function, settings::MatrixFunctionSettings)
    setIter!(A,settings)
    return MatrixFunctionProblem(T, A, f, settings)
end

abstract type MatrixFunctionState <: IterationState end
@def MatrixFunctionStatefields begin
    prob::MatrixFunctionProblem
    info::IterationInfo
    settings::MatrixFunctionSettings
end

mutable struct SchurParlettState <: MatrixFunctionState
    F::AbstractArray{<:Number,2}
    T::UpperTriangular{<:Number}
    Q::AbstractArray{<:Number,2}
    s::Number
    @MatrixFunctionStatefields
end

IterationState(prob::MatrixFunctionProblem) = MatrixFunctionState(prob, prob.settings.method)
function initFTQ(prob)
    F = zeros(Complex{prob.T}, prob.A)
    SchurF = schurfact(complex(prob.A))
    Q = SchurF[:vectors]
    T = UpperTriangular(SchurF[:T])
    @inbounds @simd for i ∈ 1:size(T,1)
        F[i,i] = prob.f(T[i,i])
    end
    return F, T, Q
end
MatrixFunctionState(prob::MatrixFunctionProblem, ::SchurParlett) =
SchurParlettState(initFTQ(prob)..., 0., prob, IterationInfo(), prob.settings)

@inline function GD.step!(state::SchurParlettState, p::Integer)
    s = state.s; T = state.T; F = state.F; n = size(state.prob.A, 1)
    @inbounds begin
        for i ∈ 1:n-p
            j::Int64 = i + p
            s = T[i,j] * (F[j,j] - F[i,i])
            for k ∈ i+1:j-1
                s += T[i,k] * F[k,j] - F[i,k] * T[k,j]
            end
            F[i,j] = s / (T[j,j] - T[i,i])
        end
    end
end

GD.errorFunction(state::MatrixFunctionState) = error("not implemented")
function GD.getResult(state::MatrixFunctionState)
    if state.prob.T <: Complex
        return state.Q * state.F * state.Q'
    else
        return map(x->x.re, state.Q * state.F * state.Q')
    end
end
@inline apply(prob::MatrixFunctionProblem) = GD.run(prob)
@inline schurParlett(A::AbstractMatrix, f::Function; kwargs...) = apply(MatrixFunctionProblem(A, f; method=SchurParlett(), kwargs...))

end
