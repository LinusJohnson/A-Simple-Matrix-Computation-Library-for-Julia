module OrthogonalizationMethods
export orthogonalize, OrthogonalizationMethod, OrthogonalizationProblem,
       Orthogonalization, SingleGramSchmidt, DoubleGramSchmidt, TripleGramSchmidt,
       ModifiedGramSchmidt
import ..GlobalDefinitions: MatrixProblem, @def

abstract type OrthogonalizationMethod end
type SingleGramSchmidt <: OrthogonalizationMethod end
type DoubleGramSchmidt <: OrthogonalizationMethod end
type TripleGramSchmidt <: OrthogonalizationMethod end
type ModifiedGramSchmidt <: OrthogonalizationMethod end
struct OrthogonalizationProblem{T <: Number} <: MatrixProblem
    Q::AbstractMatrix{T}
    w::Array{T,2}
    method::OrthogonalizationMethod
end
struct Orthogonalization{T <: Number}
    h::Union{Array{T,2}, Array{Complex{T},2}}
    β::T
    w_⟂::Union{Array{T,2}, Array{Complex{T},2}}
end

@inline function orthogonalize(prob::OrthogonalizationProblem) orthogonalize(prob, prob.method) end

@inline function orthogonalize(prob::OrthogonalizationProblem, ::SingleGramSchmidt)
    h = prob.Q'*prob.w
    y = prob.w - prob.Q*h
    β = norm(y)
    return Orthogonalization(h, β, y)
end

@inline function orthogonalize(prob::OrthogonalizationProblem, ::DoubleGramSchmidt)
    sgs = SingleGramSchmidt()
    #h1 = prob.Q'*prob.w
    #y1 = prob.w - prob.Q*h1
    #orth1 = orthogonalize(prob, sgs)
    #h2 = prob.Q'*y1
    #y2 = y1 - prob.Q*h2
    #β2 = norm(y2)
    #orth2 = orthogonalize(OrthogonalizationProblem(prob.Q, orth1.w_⟂, sgs), sgs)
    #h = orth1.h + orth2.h
    #h = h1 + h2
    #return Orthogonalization(h, β2, y2)
    #return Orthogonalization(h, orth2.β, orth2.w_⟂)

    orth1 = orthogonalize(prob, sgs)
    orth2 = orthogonalize(OrthogonalizationProblem(prob.Q, orth1.w_⟂, sgs),sgs)
    h = orth1.h + orth2.h
    return Orthogonalization(h, orth2.β, orth2.w_⟂)
end

@inline function orthogonalize(prob::OrthogonalizationProblem, ::TripleGramSchmidt)
    sgs = SingleGramSchmidt()
    orth1 = orthogonalize(prob, sgs)
    orth2 = orthogonalize(OrthogonalizationProblem(prob.Q, orth1.w_⟂, sgs),sgs)
    orth3 = orthogonalize(OrthogonalizationProblem(prob.Q, orth2.w_⟂, sgs),sgs)
    h = orth1.h + orth2.h + orth3.h
    return Orthogonalization(h, orth3.β, orth3.w_⟂)
end

@inline function orthogonalize(prob::OrthogonalizationProblem, ::ModifiedGramSchmidt)
    y::typeof(prob.w) = prob.w
    h::typeof(prob.w) = typeof(prob.w)(size(prob.Q, 2),1)
    for i_ = 1:size(prob.Q, 2)
        q = view(prob.Q, :, i_)
        h[i_] = dot(q, y)
        y -= h[i_] * q
    end
    β = norm(y)
    return Orthogonalization(h, β, y)
end

end
