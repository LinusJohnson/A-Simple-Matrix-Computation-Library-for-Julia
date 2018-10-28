module KrylovProjectionMethods
export project, KrylovProjectionProblem, KrylovProjection, ArnoldiMethod,
       KrylovProjectionMethod, ArnoldiState, ArnoldiProjectionProblem,
       KrylovProjectionSettings, OrthError, ArnoldiSettings, ProjState
using ..OrthogonalizationMethods
using ..GlobalDefinitions
const GD = GlobalDefinitions
using Parameters
import Base: start, next, done

struct ArnoldiMethod <: KrylovProjectionMethod
    orthMethod::OrthogonalizationMethod
end
const METHOD_CLASSES = [ArnoldiMethod]
getMethodClass(method::KrylovProjectionMethod) = GD.getMethodClass(method, METHOD_CLASSES)
const DEFAULT_NUM_STEPS = 50
const DEFAULT_ERR = 1e-10
const DEFAULT_METHOD = ArnoldiMethod(DoubleGramSchmidt())
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

@with_kw type ArnoldiSettings <: KrylovProjectionSettings
    @StandardIterationSettingsFields
    shift::Number
    invert::Bool
end

const settingsMap = Dict(ArnoldiMethod=>ArnoldiSettings)

function KrylovProjectionSettings(;
        maxIter::Union{Integer,Void}=nothing, maxErr::Union{AbstractFloat,Void}=nothing,
        method::Union{KrylovProjectionMethod,Void}=nothing, errFun::Union{ErrFun,Void}=nothing,
        storeIterates::Union{Bool,Void}=nothing, time::Union{Bool,Void}=nothing,
        verbose::Union{Int16,Void}=nothing,
        shift::Union{<:Number,Void}=nothing,
        invert::Union{Bool,Void}=nothing)
    params = Dict{Symbol,Any}([])
    setStandardIterationSettingsFields!(params, DEFAULT_ITER_SETTINGS, maxIter,
                                        maxErr, method, errFun, storeIterates, time, verbose)
    settingsType = settingsMap[getMethodClass(params[:method])]
    if settingsType == ArnoldiSettings
        if shift == nothing
            params[:shift] = 0
        else params[:shift] = shift end
        if invert == nothing
            params[:invert] = false
        else params[:invert] = invert end
    end
    return settingsType(;params...)
end

#function KrylovProjectionSettings(settings::IterationSettings)
#    params = struct2dict(settings)
#    filter!((k, v) -> k in fieldnames(KrylovProjectionSettings), params)
#    return KrylovProjectionSettings(;params...)
#end

function ArnoldiSettings(settings::IterationSettings)
    params = struct2dict(settings)
    filter!((k, v) -> k in fieldnames(ArnoldiSettings), params)
    return KrylovProjectionSettings(;params...)
end

function ArnoldiSettings(settings::EigValSettings)
    params = struct2dict(settings)
    filter!((k, v) -> k in fieldnames(ArnoldiSettings), params)
    params[:method] = settings.method.arnoldiMethod
    params[:storeIterates] = false
    params[:maxErr] = nothing; params[:errFun] = nothing
    params[:time] = nothing;
    for key in keys(params)
        if isa(params[key], Nullable)
            params[key] = isnull(params[key]) ? nothing : get(params[key])
        end
    end
    sett = KrylovProjectionSettings(;params...)
    sett.maxIter = params[:maxIter]
    return sett
end

function ArnoldiSettings(settings::IterationSettings,
    shift::Number,
    invert::Bool)
    params = struct2dict(settings)
    params[:shift] = shift;params[:invert] = invert
    return KrylovProjectionSettings(;params...)
end

struct KrylovProjectionProblem{T <: Number} <: MatrixProblem
    T::Type
    A::Union{AbstractMatrix{T}, AbstractArray{T}}
    b::Array{T,2}
    settings::KrylovProjectionSettings
end

function initA(A, settings)
    if settings.verbose >= 1
        settings.invert ? println("Inverting") : println("Not Inverting")
        println("shift: $(settings.shift)")
    end
    return settings.invert ? inv(A - settings.shift*eye(A)) : A - settings.shift*eye(A)
end

function KrylovProjectionProblem{T<:Number}(A::AbstractMatrix, b::Array{T,2}; kwargs...)
    settings =  KrylovProjectionSettings(;kwargs...)
    return KrylovProjectionProblem(T, initA(A, settings), b, settings)
end
KrylovProjectionProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2},
                                   settings::KrylovProjectionSettings) =
KrylovProjectionProblem(T, initA(A,settings), b, settings)

abstract type ProjState <: IterationState end
@def ProjStatefields begin
    prob::KrylovProjectionProblem{T}
    info::IterationInfo
    settings::KrylovProjectionSettings
end

mutable struct ArnoldiState{T <: Number} <: ProjState
    Q::AbstractMatrix{T}
    H::AbstractMatrix{T}
    @ProjStatefields
end

struct KrylovProjection{T <: Number}
    Q::AbstractMatrix{T} # basis
    H::AbstractMatrix{T} # Hessenberg matrix
end

IterationState(prob::KrylovProjectionProblem) = ProjState(prob,prob.settings.method)
@inline InitQ(prob::KrylovProjectionProblem) =
setindex!(zeros(eltype(prob.A), length(prob.b), get(prob.settings.maxIter,0)+1), prob.b / norm(prob.b), :, 1)
@inline InitH(prob::KrylovProjectionProblem) =
zeros(eltype(prob.A), get(prob.settings.maxIter,1)+1, get(prob.settings.maxIter,1))

ProjState(prob::KrylovProjectionProblem, method::ArnoldiMethod) =
ArnoldiState(InitQ(prob), InitH(prob), prob, IterationInfo(), prob.settings)


#Arnoldi(prob::KrylovProjectionProblem, m::Integer) = prob.settings. ?
#Arnoldi(prob,
#    setindex!(zeros(eltype(prob.A), length(prob.b), m+1), prob.b / norm(prob.b), :, 1),
#    zeros(eltype(prob.A), m+1, m), m, zeros(prob.T,m+1), zeros(Float64,m+1)) :
#Arnoldi(prob,
#    setindex!(zeros(eltype(prob.A), length(prob.b), m+1), prob.b / norm(prob.b), :, 1),
#    zeros(eltype(prob.A), m+1, m), m, Vector{prob.T}(0), Vector{Float64}(0))

#Arnoldi(prob::MatrixProblem, β::Number, m::Integer) =
#Arnoldi(KrylovProjectionProblem(prob.A, prob.b, prob.projMethod, prob.orthMethod),
#        setindex!(zeros(eltype(prob.A), length(prob.b), m+1), prob.b / β, :, 1),
#        zeros(eltype(prob.A), m+1, m), m, Vector{prob.T}(0), Vector{Float64}(0))

#function KrylovProjection(finalState::ProjState)
#    if finalState.prob.calcOrthError KrylovProjectionWithResAndError
#        KrylovProjectionWithResAndError(finalState.Q, finalState.H[1:finalState.m,:], finalState.orthErr, finalState.time)
#    else
#        KrylovProjectionNoResOrError(finalState.Q, finalState.H[1:finalState.m,:])
#    end
#end

#struct KrylovProjectionWithResAndError{T <: Number} <: KrylovProjection
#    @KrylovProjectionFields
#    orthErr::Vector{T}
#    time::Vector{Float64}
#end

#start(::ArnoldiState) = 1; done(state::ArnoldiState, k::Integer) = k > state.m

function GD.step!(state::ArnoldiState, k::Integer)
    Qk = view(state.Q, :, k:k)
    w = zeros(Qk)
    A_mul_B!(w, (state.prob.A), view(state.Q, :, k:k)) # Matrix-vector product with last element
    ### Orthogonalize w against columns of Q
    Qw⟂::Orthogonalization = orthogonalize(
            OrthogonalizationProblem(view(state.Q, :, 1:k), w, state.prob.settings.method.orthMethod))
    #println(state.prob.settings.maxIter)
    #println(typeof(state.prob.settings))
    if !isnull(state.prob.settings.maxIter)
        state.H[1:(k+1),k] = [Qw⟂.h; Qw⟂.β] # Put Gram-Schmidt coefficients into H
        state.Q[:,k+1] = Qw⟂.w_⟂ / Qw⟂.β # normalize
    else
        if k == 1
            state.H[1:(k+1),k] = [Qw⟂.h; Qw⟂.β]
        else
            state.H = cat(1, state.H, zeros(1,k-1))
            state.H = cat(2, state.H, [Qw⟂.h; Qw⟂.β])
        end
        state.Q = cat(2, state.Q, Qw⟂.w_⟂ / Qw⟂.β)
    end
    return nothing, k + 1
end

type OrthError <: ErrFun end
function GD.errorFunction(state::ProjState, ::OrthError)
    k = state.info.k[end]
    Qk = view(state.Q,:,1:k)
    return norm(Qk'*Qk - eye(k))
end

GD.getResult(state::ProjState) = KrylovProjection(state.Q,state.H)
project(prob::KrylovProjectionProblem) = GD.run(prob)
#function project(prob::KrylovProjectionProblem, method::ArnoldiMethod, m::Integer)
    # A simple implementation of the Arnoldi method.
    # The algorithm will return an Arnoldi "factorization":
    #   Q*H(1:m+1,1:m)-A*Q(:,1:m)=0
    # where Q is an orthogonal basis of the Krylov subspace
    # and H a Hessenberg matrix.
    #arnIter = Arnoldi(prob, m)
    #for i = arnIter end
    #for k = 1:m projection_step!(method, Q, H, prob.A, prob.orthMethod, k) end
    #assert(norm(arnIter.Q * arnIter.H - prob.A * view(arnIter.Q, :, 1:m)) < 1)
    #assert(norm(arnIter.Q' * arnIter.Q - eye(m+1)) < 1)
    #return KrylovProjection(arnIter)
#end

end
