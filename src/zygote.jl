
using .Zygote
using .Zygote: @adjoint, _zero, forward

#===== mapcols, maprows, MapCols =====#

@adjoint mapcols(f::Function, M::AbstractMatrix, args...) =
    ∇mapcols([ forward(x -> surevec(f(x, args...)), col) for col in eachcol(M) ], args)

@adjoint maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows([ forward(x -> surevec(f(x, args...)), row) for row in eachrow(M) ], args)

@adjoint _MapCols(f::Function, M::Matrix, dval, args...) = ∇MapCols(f, M, dval, args...)

#===== TensorCast =====#

@adjoint function TensorCast.sliceview(A::AbstractArray, code::Tuple)
    TensorCast.sliceview(A, code), Δ -> begin
        dA = _zero(A)
        foreach(copyto!, TensorCast.sliceview(dA, code), Δ)
        (dA, nothing)
    end
end

@adjoint function TensorCast.red_glue(A::AbstractArray, code::Tuple)
    TensorCast.red_glue(A, code), Δ -> (TensorCast.sliceview(Δ, code), nothing)
end

@adjoint function TensorCast.copy_glue(A::AbstractArray, code::Tuple)
    TensorCast.copy_glue(A, code), Δ -> (TensorCast.sliceview(Δ, code), nothing)
end

#===== Misc Base =====#

@adjoint function Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector})
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)),)
end

# Zygote doesn't understand views, but easy to fix:
# https://github.com/FluxML/Zygote.jl/issues/52
# now https://github.com/FluxML/Zygote.jl/pull/219

@adjoint function view(x::AbstractArray, inds...; kwargs...)
    view(x, inds...; kwargs...), dy -> begin
        dx = _zero(x)
        copyto!(view(dx, inds...; kwargs...), dy)
        (dx, map(_->nothing, inds)...)
    end
end

#===== Misc experiments =====#

@adjoint function gluecol(V::AbstractVector)
    gluecol(V), ΔM -> (collect(eachcol(ΔM)),) # does work!
end

@adjoint function collecteachcol(x)
    collecteachcol(x), dy -> begin
        dx = _zero(x)
        foreach(copyto!, collecteachcol(dx), dy)
        (dx,)
    end
end
