module TransformationMethods
export transform, TransformationProblem, HessenbergReduction,
       TransformationMethod, HessenbergReductionState, HessRed!,
       TransformationSettings, AbsMaxLowerTriErrFun, TransformationState,QR!
using ..OrthogonalizationMethods
using ..GlobalDefinitions
const GD = GlobalDefinitions
using Parameters
using Compat
import Base: start, next, done, llvmcall

type HessenbergReduction <: TransformationMethod end
type HessenbergQR <: TransformationMethod end

const METHOD_CLASSES = [HessenbergReduction, HessenbergQR]
getMethodClass(method::TransformationMethod) = GD.getMethodClass(method, METHOD_CLASSES)
const DEFAULT_NUM_STEPS = nothing
const DEFAULT_ERR = nothing
const DEFAULT_METHOD = HessenbergReduction()
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

@with_kw type HessenbergSettings <: TransformationSettings
    @StandardIterationSettingsFields
end

@with_kw type QRSettings <: TransformationSettings
    @StandardIterationSettingsFields
end

const settingsMap = Dict(HessenbergReduction=>HessenbergSettings, HessenbergQR=>QRSettings)

function TransformationSettings(;
        maxIter::Union{Integer,Void}=nothing, maxErr::Union{AbstractFloat,Void}=nothing,
        method::Union{TransformationMethod,Void}=nothing, errFun::Union{ErrFun,Void}=nothing,
        storeIterates::Union{Bool,Void}=nothing, time::Union{Bool,Void}=nothing,
        verbose::Union{Int16,Void}=nothing)
    params = Dict{Symbol,Any}([])
    setStandardIterationSettingsFields!(params, DEFAULT_ITER_SETTINGS, maxIter,
                                        maxErr, method, errFun, storeIterates, time, verbose)
    settingsType = settingsMap[getMethodClass(params[:method])]
    return settingsType(;params...)
end

struct TransformationProblem <: MatrixProblem
    T::Type
    A::AbstractMatrix{<:Number}
    settings::TransformationSettings
end
setIter!(A::AbstractMatrix, settings::TransformationSettings) =
setIter!(A, settings, settings.method)
function setIter!(A::AbstractMatrix, settings::TransformationSettings, ::HessenbergReduction)
    settings.maxIter = size(A,1) - 2
    settings.maxErr = nothing
end
function setIter!(A::AbstractMatrix, settings::TransformationSettings, ::HessenbergQR)
    #settings.maxIter = size(A,1)*20
end

function TransformationProblem{T<:Number}(A::AbstractMatrix{T}; kwargs...)
    settings = TransformationSettings(;kwargs...)
    setIter!(A,settings)
    return TransformationProblem(T, A, settings)
end
function TransformationProblem(A::AbstractMatrix{<:Number}, settings::TransformationSettings)
    setIter!(A,settings)
    return TransformationProblem(T, A, settings)
end

abstract type TransformationState <: IterationState end
@def TransformationStatefields begin
    prob::TransformationProblem
    info::IterationInfo
    settings::TransformationSettings
end

mutable struct HessenbergReductionState <: TransformationState
    C::Array{<:Number,2}
    u::Array{<:Number,2}
    uu_2::Array{<:Number,2}
    β::Float64
    @TransformationStatefields
end

mutable struct HessenbergQRState <: TransformationState
    C::Array{<:Number,2}
    S::Array{<:Number,2}
    @TransformationStatefields
end

IterationState(prob::TransformationProblem) = TransformationState(prob, prob.settings.method)
initC(prob) = zeros(size(prob.A,1), size(prob.A,2))
initu(prob) = zeros(size(prob.A,1)-1, 1)
inituu_2(prob) = zeros(size(prob.A,1)-1, size(prob.A,1)-1)
TransformationState(prob::TransformationProblem, method::HessenbergReduction) =
HessenbergReductionState(initC(prob), initu(prob), inituu_2(prob), 0., prob, IterationInfo(), prob.settings)

TransformationState(prob::TransformationProblem, method::HessenbergQR) =
HessenbergQRState(zeros(size(prob.A,1),1), zeros(size(prob.A,1),1), prob, IterationInfo(), prob.settings)

@inline function GD.step!(state::HessenbergQRState, k::Integer)
    @inbounds begin
        n::Int64 = size(state.prob.A, 1)
        for i = 1:n-1
            a = state.prob.A[i,i]
            b = state.prob.A[i+1,i]
            state.C[i], state.S[i], _ = LinAlg.givensAlgorithm(a,b)
            @simd for j = i:n
                a1 = state.prob.A[i,j]
                a2 = state.prob.A[i+1,j]
                state.prob.A[i,j] = state.C[i]*a1 + state.S[i]*a2
                state.prob.A[i+1,j] = -state.S[i]'*a1 + state.C[i]*a2
            end
        end
        for i = 1:n-1
            @simd for j = 1:i+1
                a1 = state.prob.A[j,i]
                a2 = state.prob.A[j,i+1]
                state.prob.A[j,i] = state.C[i]*a1 + state.S[i]'*a2
                state.prob.A[j,i+1] = -state.S[i]*a1 + state.C[i]*a2
            end
        end
    end
end

@inline function GD.step!(state::HessenbergReductionState, k::Integer)
    # See http://www.netlib.org/blas/blasqr.pdf
    @inbounds begin
        n::Int64 = size(state.prob.A, 1)
        uₖ = view(state.u, 1:n-k, 1)

        Aₖ = view(state.prob.A, k+1:n, k:n)
        copyto!(uₖ, 1, Aₖ, 1, n-k)
        state.β = norm(uₖ)
        uₖ[1] -= state.β
        normalize!(uₖ)
        uₖuₖ_2 = view(state.uu_2, 1:n-k, 1:n-k) # is zero
        BLAS.ger!(2., uₖ, uₖ, uₖuₖ_2) # do 2*uₖ*uₖ'
        # construct (I - 2*uₖ*uₖ'), (uₖuₖ_2 is square)
        uₖuₖ_2 .*= -1
        uₖuₖ_2_diag = view(state.uu_2, 1:n:(n-k)*n)
        uₖuₖ_2_diag .+= 1
        #A = (I- uₖuₖ_2)A
        Cₖ = view(state.C, 1:n-k, 1:n-k+1)
        BLAS.gemm!('N','N', 1., uₖuₖ_2, Aₖ, 0., Cₖ)
        copyto!(Aₖ, Cₖ)

        #A = A(I- uₖuₖ_2)
        Cₖ = view(state.C, 1:n, 1:n-k)
        Aₖ = view(state.prob.A, 1:n, k+1:n)
        BLAS.gemm!('N', 'N', 1., Aₖ, uₖuₖ_2, 0., Cₖ)
        copyto!(Aₖ, Cₖ)
        uₖuₖ_2 .= 0 # ensure it's zero for next loop
    end
    return nothing, k + 1
end

type AbsMaxLowerTriErrFun <: ErrFun end
GD.errorFunction(state::HessenbergReductionState, ::AbsMaxLowerTriErrFun) = maximum(abs(tril(state.Aₖ,-1)))
GD.getResult(state::TransformationState) = nothing
@inline transform(prob::TransformationProblem) = GD.run(prob)[2]
@inline HessRed!(A::AbstractMatrix; kwargs...) = transform(TransformationProblem(A; method=HessenbergReduction(), kwargs...))
@inline QR!(A::AbstractMatrix; kwargs...) = transform(TransformationProblem(A; method=HessenbergQR(), kwargs...))

end
