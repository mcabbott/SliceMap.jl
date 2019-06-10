
using .Zygote
using .Zygote: @adjoint, _zero, forward

#===== mapcols, maprows, MapCols =====#

@adjoint _mapcols(map::Function, f::Function, M::AbstractMatrix, args...) =
    ∇mapcols(map, map(col -> forward(x -> surevec(f(x, args...)), col), eachcol(M)), args)

@adjoint maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows(map(row -> forward(x -> surevec(f(x, args...)), row), eachrow(M)), args)

@adjoint _MapCols(map::Function, f::Function, M::Matrix, dval, args...) =
    ∇MapCols(map, f, M, dval, args...)

#===== TensorCast =====#

@adjoint TensorCast.sliceview(A::AbstractArray, code::Tuple) =
    TensorCast.sliceview(A, code), Δ -> (TensorCast.glue(Δ, code), nothing)

@adjoint TensorCast.red_glue(A::AbstractArray, code::Tuple) =
    TensorCast.red_glue(A, code), Δ -> (TensorCast.sliceview(Δ, code), nothing)

@adjoint TensorCast.copy_glue(A::AbstractArray, code::Tuple) =
    TensorCast.copy_glue(A, code), Δ -> (TensorCast.sliceview(Δ, code), nothing)

#===== JuliennedArrays =====#

@adjoint JuliennedArrays.Slices(whole, along...) =
    Slices(whole, along...), Δ -> (Align(Δ, along...), map(_->nothing, along)...)

@adjoint JuliennedArrays.Align(whole, along...) =
    Align(whole, along...), Δ -> (Slices(Δ, along...), map(_->nothing, along)...)

#===== Misc Base =====#

@adjoint Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector}) =
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)),)

# https://github.com/FluxML/Zygote.jl/pull/219
@adjoint function view(x::AbstractArray, inds...; kwargs...)
    view(x, inds...; kwargs...), dy -> begin
        dx = _zero(x)
        copyto!(view(dx, inds...; kwargs...), dy)
        (dx, map(_->nothing, inds)...)
    end
end

#===== Misc experiments =====#

@adjoint gluecol(V::AbstractVector) =
    gluecol(V), ΔM -> (collect(eachcol(ΔM)),) # does work!

@adjoint function collecteachcol(x)
    collecteachcol(x), dy -> begin
        dx = _zero(x)
        foreach(copyto!, collecteachcol(dx), dy)
        (dx,)
    end
end
