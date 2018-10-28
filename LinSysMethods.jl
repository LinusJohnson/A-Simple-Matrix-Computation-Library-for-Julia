module LinSysMethods
export solve, GMRESMethod, CGMethod, CGNEMethod, LinSysProblem, LinSysSol
using ..KrylovProjectionMethods
using ..OrthogonalizationMethods
import Base: start, next, done
import ..GlobalDefinitions: MatrixProblem, @def

abstract type LinSysMethod end
type GMRESMethod <: LinSysMethod end
type CGMethod <: LinSysMethod end
type CGNEMethod <: LinSysMethod end
abstract type LinSysProblem{T <: Number} <: MatrixProblem end
@def LinSysProblemfields begin
    T::Type
    A::AbstractMatrix{T}
    b::Array{T,2}
    method::LinSysMethod
    sol::Nullable{Array{T,2}} # solution for error
    calcResnorm::Bool
    calcErr::Bool
end

struct GMRESLinSysProblem{T <: Number} <: LinSysProblem{T}
    @LinSysProblemfields
    projMethod::KrylovProjectionMethod
    orthMethod::OrthogonalizationMethod
end
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2}, method::GMRESMethod,
projMethod::KrylovProjectionMethod, orthMethod::OrthogonalizationMethod, sol::Array{T,2}) =
GMRESLinSysProblem(T, A, b, method, Nullable(sol), true, true, projMethod, orthMethod)
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2}, method::GMRESMethod,
projMethod::KrylovProjectionMethod, orthMethod::OrthogonalizationMethod) =
GMRESLinSysProblem(T, A, b, method, Nullable{Array{T,2}}(), false, false, projMethod, orthMethod)
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2}, method::GMRESMethod) =
GMRESLinSysProblem(T, A, b, method, Nullable{Array{T,2}}(), false, false, ArnoldiMethod(), DoubleGramSchmidt())
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2}) =
GMRESLinSysProblem(T, A, b, GMRESMethod(), Nullable{Array{T,2}}(), false, false, ArnoldiMethod(), DoubleGramSchmidt())
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2}, sol::Array{T,2}) =
GMRESLinSysProblem(T, A, b, GMRESMethod(), Nullable(sol), true, true, ArnoldiMethod(), DoubleGramSchmidt())
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2},  method::GMRESMethod, sol::Array{T,2}) =
GMRESLinSysProblem(T, A, b, method, Nullable(sol), true, true, ArnoldiMethod(), DoubleGramSchmidt())
LinSysProblem{T<:Number}(A::AbstractMatrix{T}, b::Array{T,2},  method::GMRESMethod, calcRes::Bool) =
GMRESLinSysProblem(T, A, b, method, Nullable{Array{T,2}}(), calcRes, false, ArnoldiMethod(), DoubleGramSchmidt())

const CGLikeMethod = Union{CGMethod, CGNEMethod}
struct CGLinSysProblem{T <: Number} <: LinSysProblem{T}
    @LinSysProblemfields
end
LinSysProblem{T<:Number}(A::AbstractMatrix{}, b::Array{T,2}, method::CGLikeMethod, sol::Array{T,2}) =
CGLinSysProblem(T, A, b, method, Nullable(sol), true, true)
LinSysProblem{T<:Number}(A::AbstractMatrix{}, b::Array{T,2}, method::CGLikeMethod) =
CGLinSysProblem(T, A, b, method, Nullable{Array{T,2}}(), false, false)
LinSysProblem{T<:Number}(A::AbstractMatrix{}, b::Array{T,2}, method::CGLikeMethod, calcRes::Bool) =
CGLinSysProblem(T, A, b, method, Nullable{Array{T,2}}(), calcRes, false)

abstract type LinSysState end
@def LinSysStatefields begin
    T::Type
    x::Array{T,2}
    prob::LinSysProblem
    m::Int
    resnorm::Vector{T}
    err::Vector{Nullable{T}}
    time::Vector{Float64}
end

mutable struct GMRES{T} <: LinSysState
    @LinSysStatefields
    arnoldiState::ArnoldiState
    β::Union{T,Float64}
end
GMRES(x₀::Array, prob::LinSysProblem, β, m::Integer) =
GMRES(prob.T, x₀, prob, m, Vector{prob.T}(m+1), Vector{Nullable{prob.T}}(m+1), Vector{Float64}(m+1), ArnoldiState(prob, β, m), β)

mutable struct CG{T<:Number} <: LinSysState
    @LinSysStatefields
    p::Array{T,2}
    r::Array{T,2}
    storeIterates::Bool
    iterates::Nullable{AbstractMatrix{T}}
end
CG(x₀::Array, prob::LinSysProblem, r, m::Integer, storeIterates::Bool) = storeIterates ?
CG(prob.T, x₀, prob, m, Vector{prob.T}(m+1), Vector{Nullable{prob.T}}(m+1), Vector{Float64}(m+1), r, r, true, Nullable(AbstractMatrix{prob.T}(size(x,1), m))) :
CG(prob.T, x₀, prob, m, Vector{prob.T}(m+1), Vector{Nullable{prob.T}}(m+1), Vector{Float64}(m+1), r, r, false, Nullable{AbstractMatrix{prob.T}}())

abstract type LinSysSol end
function LinSysSol(finalState::LinSysState)
    resnorm = Vector{finalState.prob.T}(); err = Vector{finalState.prob.T}()
    if finalState.prob.calcResnorm
        resnorm = finalState.resnorm
    end
    if finalState.prob.calcErr
         err = map(get, finalState.err)
    end
    if finalState.prob.calcResnorm || finalState.prob.calcErr
        LinSysSolWithResAndError(finalState.x, resnorm, err, finalState.time)
    else
        LinSysSolNoResOrError(finalState.x)
    end
end
struct LinSysSolNoResOrError{T <: Number} <: LinSysSol
    x::Array{T,2}
end
struct LinSysSolWithResAndError{T <: Number} <: LinSysSol
    x::Array{T,2}
    res::Vector{T}
    err::Vector{T}
    time::Vector{Float64}
end

function calc_res!(x, state::LinSysState, k::Integer)
    state.resnorm[k] = norm(state.prob.A*x - state.prob.b)
end

function calc_err!(x, state::LinSysState, k::Integer)
    state.err[k] = map(sol -> norm(x - sol), state.prob.sol)
end

function start(state::LinSysState)
    state.time[1] = 0;
    if state.prob.calcResnorm calc_res!(state.x, state, 1) end
    if state.prob.calcErr calc_err!(state.x, state, 1) end
    return 1
end

function next(state::LinSysState, k::Integer)
    tic()
    step!(state, state.prob.method, k)
    if state.prob.calcResnorm || state.prob.calcResnorm
        x = get_curr_est(state, k)
        if state.prob.calcResnorm calc_res!(x, state, k+1) end
        if state.prob.calcErr calc_err!(x, state, k+1) end
    end
    state.time[k+1] = toq() + state.time[k]
    return nothing, k+1
end

function step!(state::GMRES, method::GMRESMethod, k::Integer)
    next(state.arnoldiState, k)
end

get_curr_est(state::GMRES, k::Integer) =
state.arnoldiState.Q[:,1:k-1] * (state.arnoldiState.H[1:k,1:k-1] \
setindex!(zeros(eltype(state.prob.A), k, 1), state.β, 1, 1))

function done(state::GMRES, k::Integer)
    if k > state.m
        state.x = get_curr_est(state, k)
        return true
    end
    return false
end

function solve(prob::LinSysProblem, n::Int) solve(prob, prob.method, n) end

function solve(prob::GMRESLinSysProblem, method::GMRESMethod, m::Int)
    GMRESIter = GMRES(zeros(prob.b), prob, norm(prob.b), m)
    for m = GMRESIter end
    return LinSysSol(GMRESIter)
end

@def CGLikeStep begin
    rₖᵀrₖ = state.r.' * state.r
    αₖ = rₖᵀrₖ / (state.p.' * Apₖ)
    state.x += αₖ.*state.p
    if state.storeIterates state.iterates[k] = state.x end
    state.r -= αₖ.*Apₖ
    βₖ = state.r.' * state.r / rₖᵀrₖ
    state.p = state.r + βₖ.*state.p
end

function step!(state::CG, method::CGMethod, k::Integer)
    Apₖ = state.prob.A * state.p
    @CGLikeStep
end

function step!(state::CG, method::CGNEMethod, k::Integer)
    Apₖ = state.prob.A.' * (state.prob.A * state.p)
    @CGLikeStep
end

get_curr_est(state::CG, k::Integer) = state.x
done(state::CG, k::Integer) = k > state.m

function solve(prob::CGLinSysProblem, method::CGMethod, m::Int, storeIterates::Bool = false)
    x₀ = zeros(prob.b)
    CGIter = CG(x₀, prob, prob.b, m, storeIterates)
    for m = CGIter end
    return CGIter.storeIterates ? (LinSysSol(CGIter), CGIter.iterates) : LinSysSol(CGIter)
end

function solve(prob::CGLinSysProblem, method::CGNEMethod, m::Int, storeIterates::Bool = false)
    x₀ = zeros(prob.b)
    CGNEIter = CG(x₀, prob, prob.A.'*prob.b, m, storeIterates)
    for m = CGNEIter end
    return CGNEIter.storeIterates ? (LinSysSol(CGNEIter), CGNEIter.iterates) : LinSysSol(CGNEIter)
end
end
