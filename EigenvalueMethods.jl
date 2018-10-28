module EigenvalueMethods
export EigValProblem, OrthProblem, EigenPair, PowerMethod, ArnoldiIteration,
       RayleighQuotientIteration, QRMethod, getEigApprox, errorPlot, EigValSettings,
       ExactλErrFun, AbsMaxLowerTriErrFun, genPowMatrix, ExactλErrNormFun
using ..OrthogonalizationMethods
using ..KrylovProjectionMethods
using ..GlobalDefinitions
const GD = GlobalDefinitions
using Parameters
import Base: start, next, done

abstract type KrylovMethod <: EigMethod end
abstract type PowerMethodLikeMethod <: KrylovMethod end
type PowerMethod <: PowerMethodLikeMethod end
type RayleighQuotientIteration <: PowerMethodLikeMethod end
struct ArnoldiIteration <: KrylovMethod
    arnoldiMethod::KrylovProjectionMethod
end
ArnoldiIteration(orthMethod::OrthogonalizationMethod) =
ArnoldiIteration(ArnoldiMethod(orthMethod))

type QRMethod <: EigMethod end

const METHOD_CLASSES = [PowerMethodLikeMethod, ArnoldiIteration, QRMethod]
getMethodClass(method::EigMethod) = GD.getMethodClass(method, METHOD_CLASSES)
const DEFAULT_NUM_STEPS = 50
const DEFAULT_ERR = 1e-10
const DEFAULT_METHOD = QRMethod()
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

@def ExtraEigValSettingsFields begin
    exactλ::Nullable{Array{<:Number,2}}
end

@with_kw type QRMethodsEigValSettings <: EigValSettings
    @ExtraEigValSettingsFields
    @StandardIterationSettingsFields
end

@def ExtraKrylovMethodsEigValSettings begin
    startVec::Array{<:Number,2}
end

@with_kw type PowerMethodLikeMethodEigValSettings <: EigValSettings
    @ExtraEigValSettingsFields
    @StandardIterationSettingsFields
    @ExtraKrylovMethodsEigValSettings
end

const DEFAULT_SHIFT = 0
const DEFAULT_INVERT = false
@with_kw type ArnoldiMethodEigValSettings <: EigValSettings
    @ExtraEigValSettingsFields
    @StandardIterationSettingsFields
    @ExtraKrylovMethodsEigValSettings
    shift::Number
    invert::Bool
end

const settingsMap = Dict(PowerMethodLikeMethod=>PowerMethodLikeMethodEigValSettings,
                         ArnoldiIteration=>ArnoldiMethodEigValSettings,
                         QRMethod=>QRMethodsEigValSettings)

function EigValSettings(;
        maxIter::Union{Integer,Void}=nothing, maxErr::Union{AbstractFloat,Void}=nothing,
        method::Union{EigMethod,Void}=nothing, errFun::Union{ErrFun,Void}=nothing,
        storeIterates::Union{Bool,Void}=nothing, time::Union{Bool,Void}=nothing,
        verbose::Union{Int16,Void}=nothing,
        startVec::Union{Array{<:Number,2},Void}=nothing,
        exactλ::Union{Array{<:Number,2},Void}=nothing,
        shift::Union{<:Number,Void}=nothing,
        invert::Union{Bool,Void}=nothing)
    params = Dict{Symbol,Any}([])
    setStandardIterationSettingsFields!(params, DEFAULT_ITER_SETTINGS, maxIter,
                                        maxErr, method, errFun, storeIterates, time, verbose)
    methodClass = getMethodClass(params[:method])
    settingsType = settingsMap[methodClass]
    if methodClass in subtypes(KrylovMethod)
        if startVec == nothing
            error("No initial vector provided")
        else params[:startVec] = startVec end
    end
    if exactλ == nothing
        params[:exactλ] = Nullable{Array{<:Number,2}}()
    else params[:exactλ] = exactλ end
    if methodClass == ArnoldiIteration
        if shift == nothing
            params[:shift] = DEFAULT_SHIFT
        else params[:shift] = shift end
        if invert == nothing
            params[:invert] = DEFAULT_INVERT
        else params[:invert] = invert end
    end
    return settingsType(;params...)
end

struct EigValProblem{T <: Number} <: MatrixProblem
    T::Type
    A::AbstractMatrix{T}
    settings::EigValSettings
end

struct EigenPair{T<:Number}
    λ::T
    w::Array{T,2}
end

EigValProblem{T<:Number}(A::AbstractMatrix{T}; kwargs...) =
EigValProblem(T::Type, A::AbstractMatrix, EigValSettings(;kwargs...))
EigValProblem{T<:Number}(A::AbstractMatrix{T}, settings::EigValSettings) =
EigValProblem(T::Type, A::AbstractMatrix, settings::EigValSettings)

abstract type EigMethodState <: IterationState end
@def EigMethodStateFields begin
    prob::EigValProblem
    info::IterationInfo
    settings::EigValSettings
end

type QRState{T <: Number} <: EigMethodState
    Aₖ::AbstractMatrix{T}
    Uₖ::AbstractMatrix{T}
    Qₖ::AbstractMatrix{T}
    Rₖ::AbstractMatrix{T}
    @EigMethodStateFields
end
IterationState(prob::EigValProblem) = EigMethodState(prob, prob.settings.method)
EigMethodState(prob::EigValProblem, method::QRMethod) =
QRState(prob.A, eye(prob.A), eye(prob.A), zeros(prob.A), prob,
        IterationInfo(), prob.settings)

type PowerMethodState{T <: Number} <: EigMethodState
    vₖ::Array{T,2}
    wₖ::Array{T,2}
    λₖ::T
    @EigMethodStateFields
end

EigMethodState(prob::EigValProblem, method::PowerMethod) =
PowerMethodState(prob.settings.startVec, Array{prob.T,2}(size(prob.settings.startVec)),
                 0., prob, IterationInfo(), prob.settings)

EigMethodState(prob::EigValProblem, method::RayleighQuotientIteration) =
PowerMethodState(prob.settings.startVec,
    Array{prob.T,2}(size(prob.settings.startVec)),
    dot(prob.settings.startVec.', prob.A * prob.settings.startVec),
    prob, IterationInfo(), prob.settings)

type ArnoldiIterationState <: EigMethodState
    arnoldiState::ArnoldiState
    @EigMethodStateFields
end

function EigMethodState(prob::EigValProblem, method::ArnoldiIteration)
    kryProjProb::KrylovProjectionProblem =
    KrylovProjectionProblem(prob.A, Array{prob.T,2}(prob.settings.startVec),
                            ArnoldiSettings(prob.settings))
    return ArnoldiIterationState(ProjState(kryProjProb, kryProjProb.settings.method),
                                 prob,
                                 IterationInfo(),
                                 prob.settings)
end

function GD.step!(state::QRState, k::Integer)
    state.Qₖ, state.Rₖ = qr(state.Aₖ)
    state.Aₖ = state.Rₖ * state.Qₖ
    state.Uₖ *= state.Qₖ
end

function powerMethodLikeWStep!(state::PowerMethodState,::PowerMethod)
    state.wₖ = state.prob.A * state.vₖ
end

function powerMethodLikeWStep!(state::PowerMethodState,::RayleighQuotientIteration)
    state.wₖ = (state.prob.A - state.λₖ*eye(state.prob.A)) \ state.vₖ
end

function GD.step!(state::ArnoldiIterationState, k::Integer)
    GD.next(state.arnoldiState, k)
end

function GD.step!(state::PowerMethodState, k::Integer)
    powerMethodLikeWStep!(state, state.settings.method)
    state.vₖ = state.wₖ / norm(state.wₖ)
    state.λₖ = dot(state.vₖ', state.prob.A * state.vₖ)
end

getEst(state::Union{QRState,PowerMethodState}) = getλEst(state), getVEst(state)
getλEst(state::QRState) = diag(state.Aₖ)
getVEst(state::QRState) = [state.Qₖ[i:i,:] for i = 1:size(state.Aₖ, 1)]
getλEst(state::PowerMethodState) = state.λₖ
getVEst(state::PowerMethodState) = [state.vₖ]
function getEst(state::ArnoldiIterationState)
    λs, vs = eig(state.arnoldiState.H[1:state.info.k[end],1:state.info.k[end]])
    return λs, [vs[:,i:i] for i = 1:size(vs,2)]
end
getλEst(state::ArnoldiIterationState) = getEst(state)[1]
getVEst(state::ArnoldiIterationState) = getEst(state)[2]

function GD.getResult(state::EigMethodState)
    return [EigenPair(λ, v) for (λ, v) = zip(getEst(state)...)]
end

getEigApprox(prob::EigValProblem) = GD.run(prob)


# function getEigApprox(prob::EigValProblem, method::PowerMethod, n::Int)
#     v::Array{prob.T, 2} = get(prob.settings.startVec, randn(prob.T, (size(prob.A,1),1)))
#     w = typeof(v)(size(v))
#     λ::prob.T = 0
#     for i_ = 1:n
#         w = prob.A * v
#         v = w / norm(w)
#         λ = dot(v', prob.A * v)
#     end
#     return EigenPair(λ, w)
# end
#
# function getEigApprox(prob::EigValProblem, method::RayleighQuotientIteration, n::Integer)
#     v::Array{prob.T, 2} = get(prob.settings.startVec, randn(prob.T, (size(prob.A,1),1)))
#     w::typeof(v) = typeof(v)(size(v))
#     λ::prob.T = dot(v.', prob.A * v)
#     I::typeof(prob.A) = eye(prob.A)
#     for i_ = 1:n
#         w = (prob.A - λ*I) \ v
#         v = w / norm(w)
#         λ = dot(v', prob.A * v)
#     end
#     return EigenPair(λ, v)
# end

type ExactλErrFun <: ErrFun end
type ExactλErrNormFun <: ErrFun end
type AbsMaxLowerTriErrFun <: ErrFun end

GD.errorFunction(state::EigMethodState, ::ExactλErrFun) =  minimum(minimum.(abs.([λ .- get(state.settings.exactλ) for λ in getλEst(state)])))
GD.errorFunction(state::Union{ArnoldiIterationState, QRState}, ::ExactλErrNormFun) =  norm(getλEst(state) - get(state.settings.exactλ))
GD.errorFunction(state::QRState, ::AbsMaxLowerTriErrFun) = maximum(abs(tril(state.Aₖ,-1)))

function genPowMatrix(prob::EigValProblem, m::Integer)
    K_m = zeros(size(prob.settings.startVec, 1), m)
    K_m[:,1] = prob.settings.startVec
    for i = 2:m
        K_m[:,i] = prob.A * K_m[:,i-1]
    end
    for i = 2:m
        K_m[:,i] ./= norm(K_m[:,i])
    end
    return K_m
end

end
