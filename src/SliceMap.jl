
module SliceMap

export mapcols, MapCols, maprows, slicemap

using MacroTools, Tracker, Zygote, WeightedArrays
using Tracker: TrackedMatrix, track, @grad, data
using Zygote: @adjoint, _zero

#========== Reverse, Eachslice ==========#

"""
    mapcols(f, m) ≈ mapreduce(f, hcat, eachcol(m)) ≈ mapslices(f, m, dims=1)

This is a more efficient version of the functions on the right.
For `f(x::Vector)::Matrix` it reshapes like `mapslices(vec∘f, m, dims=1)`.

It provides a gradient for Tracker and Zygote, saving the backward function for each slice.

Any arguments after the matrix are passed to `f` as scalars, i.e.
`mapcols(f, m, args...) = reduce(hcat, f(col, args...) for col in eeachcol(m))`.
They do not get sliced/iterated (unlike `map`), nor are their gradients tracked.
"""
mapcols(f::Function, M::AbstractMatrix, args...) =
    reduce(hcat, [ surevec(f(col, args...)) for col in eachcol(M) ])

mapcols(f::Function, M::WeightedMatrix, args...) =
    Weighted(mapcols(f, M.array, args...), M.weights, M.opt)

surevec(x::Number) = [x] # to allow f vector -> scalar, as mapslices does
surevec(A) = vec(A)      # to allow f vector -> matrix, by reshaping

mapcols(f::Function, M::TrackedMatrix, args...) = track(mapcols, f, M, args...)

@grad mapcols(f::Function, M::AbstractMatrix, args...) =
    ∇mapcols([ Tracker.forward(x -> surevec(f(x, args...)), col) for col in eachcol(data(M)) ], args)

@adjoint mapcols(f::Function, M::AbstractMatrix, args...) =
    ∇mapcols([ Zygote.forward(x -> surevec(f(x, args...)), col) for col in eachcol(M) ], args)

function ∇mapcols(forwards, args)
    reduce(hcat, data.(first.(forwards))), Δ -> begin
        cols = [ data(last(fwd)(Δcol)[1]) for (fwd, Δcol) in zip(forwards, eachcol(data(Δ))) ]
        (nothing, reduce(hcat, cols), map(_->nothing, args)...)
    end
end

"""
    maprows(f, M) ≈ mapslices(f, M, dims=2)

Like `mapcols()` but for rows.
"""
maprows(f::Function, M::AbstractMatrix, args...) =
    reduce(vcat, [ surerow(f(col, args...)) for col in eachrow(M) ])

surerow(x) = transpose(surevec(x))

maprows(f::Function, M::TrackedMatrix, args...) = track(maprows, f, M, args...)

@grad maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows([ Tracker.forward(x -> surerow(f(x, args...)), row) for row in eachrow(data(M)) ], args)

@adjoint maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows([ Zygote.forward(x -> surerow(f(x, args...)), row) for row in eachrow(M) ], args)

function ∇maprows(forwards, args)
    reduce(vcat, data.(first.(forwards))), Δ -> begin
        rows = [ data(last(fwd)(Δrow)[1]) for (fwd, Δrow) in zip(forwards, eachrow(data(Δ))) ]
        (nothing, reduce(vcat, rows), map(_->nothing, args)...)
    end
end


#========== Forward, Static ==========#

using StaticArrays, ForwardDiff, WeightedArrays

struct MapCols{d} end

"""
    MapCols{d}(f, m::Matrix, args...)

Similar to `mapcols(f, m, args...)`, but slices `m` into `SVector{d}` columns.
Their length `d = size(M,1)` should ideally be provided for type-stability, but is not required.

The gradient for Tracker and Zygote uses `ForwardDiff` on each slice.
"""
MapCols(f::Function, M::AT, args...) where {AT<:WeightedArrays.MaybeWeightedMatrix} =
    MapCols{size(M,1)}(f, M, args...)

MapCols{d}(f::Function, M::WeightedMatrix, args...) where {d} =
    Weighted(MapCols{d}(f, M.array, args...), M.weights, M.opt)

MapCols{d}(f::Function, M::AbstractMatrix, args...) where {d} = _MapCols(f, M, Val(d), args...)

function _MapCols(f::Function, M::Matrix{T}, ::Val{d}, args...) where {T,d}
    d == size(M,1) || error("expected M with $d columns")
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(M))
    B = map(col -> surevec(f(col, args...)), A)
    reduce(hcat, B)
    # maybestaticgluecols(B)
end

# surevec(x::MArray) = Array(x) # avoid making a huge MArray, ad
# surevecS(x::Number) = @SVector [x]
# surevecS(A) = vec(A) # like surevec

function maybestaticgluecols(B)
    TB = eltype(B)
    if TB <: SArray
        C = collect(reshape(reinterpret(eltype(TB), B),:,length(B)))
    elseif TB <: MArray
        C = reduce(hcat, Array.(B))
    else
        C = reduce(hcat, B)
    end
end

_MapCols(f::Function, M::TrackedMatrix, dval, args...) = track(_MapCols, f, M, dval, args...)

@grad _MapCols(f::Function, M::TrackedMatrix, dval, args...) = ∇MapCols(f, M, dval, args...)

@adjoint _MapCols(f::Function, M::Matrix, dval, args...) = ∇MapCols(f, M, dval, args...)

function ∇MapCols(f::Function, M::AbstractMatrix{T}, dval::Val{d}, args...) where {T,d}
    d == size(M,1) || error("expected M with $d columns")
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(data(M)))

    dualcol = SVector(ntuple(j->ForwardDiff.Dual(0, ntuple(i->i==j ? 1 : 0, dval)...), dval))
    C = map(col -> surevec(f(col .+ dualcol, args...)), A)

    Z = reduce(hcat, map(col -> ForwardDiff.value.(col), C))

    function back(ΔZ)
        ∇M = zeros(eltype(data(ΔZ)), size(M))
        @inbounds for c=1:size(M,2)
            part = ForwardDiff.partials.(C[c])
            for r=1:d
                for i=1:size(ΔZ,1)
                    ∇M[r,c] += data(ΔZ)[i,c] * part[i].values[r]
                end
            end
        end
        (nothing, ∇M, nothing, map(_->nothing, args)...)
    end
    Z, back
end


#========== Gradient for eachslice / reduce ==========#

export gluecol, collecteachcol
export mapcols2, mapcols4, mapcols5, mapcols6, mapcols7

gluecol(V::AbstractVector{<:AbstractVector}) = reduce(hcat, V)

gluecol(V::AbstractVector{<:TrackedVector}) = track(gluecol, V)

@grad function gluecol(V::AbstractVector)
    gluecol(data.(V)), ΔM -> (collect(eachcol(data(ΔM))),) # doesn't work
end

Zygote.@adjoint function gluecol(V::AbstractVector)
    gluecol(V), ΔM -> (collect(eachcol(ΔM)),) # does work!
end

function mapcols2(f, A)
    cols = [A[:,c] for c=1:size(A,2)]
    res = f.(cols)
    gluecol(res)
end

# Apply that straight to reduce(hcat,...)

Zygote.@adjoint function Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector})
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)),)
end

function mapcols4(f, A)
    cols = [view(A,:,c) for c=1:size(A,2)]
    res = map(f, cols)
    reduce(hcat, res)
end

# Zygote doesn't understand views, but easy to fix:
# https://github.com/FluxML/Zygote.jl/issues/52
# now https://github.com/FluxML/Zygote.jl/pull/219

Zygote.@adjoint function view(x::AbstractArray, inds...; kwargs...)
    view(x, inds...; kwargs...), dy -> begin
        dx = _zero(x)
        copyto!(view(dx, inds...; kwargs...), dy)
        (dx, map(_->nothing, inds)...)
    end
end

# Surprisingly dy for eachcol seems to know the answer?
# typeof(dy) = NamedTuple{(:f, :iter),Tuple{NamedTuple{(:A,),Tuple{Array{Float64,2}}},Array{Nothing,1}}}
# dy = (f = (A = [47.9325 51.3781
# Which means this works... but uses as much memory as gradient of array of views:

#=Zygote.@adjoint function eachcol(x::AbstractMatrix)
    eachcol(x), dy -> (dy.f.A,) #= begin
        @show typeof(dy) dy
        dx = zero(x) .+ 0.0  # zeros(eltype(dy), size(x))
        foreach(copyto!, eachcol(dx), dy)
        (dx,)
    end =#
end=#

# @adjoint eachcol(x) = eachcol(x), dy -> (dy.f.A,)

function mapcols5(f, A)
    cols = collect(eachcol(A))
    res = map(f, cols)
    reduce(hcat, res)
end

collecteachcol(x) = collect(eachcol(x))

@adjoint function collecteachcol(x)
    collecteachcol(x), dy -> begin
        dx = _zero(x)
        foreach(copyto!, collecteachcol(dx), dy)
        (dx,)
    end
end

function mapcols6(f, A)
    cols = collecteachcol(A)
    res = map(f, cols)
    reduce(hcat, res)
end

# function mapcols7(f, A)
#     cols = eachcol(A) # without collect. Zygote.gradient -> StackOverflowError
#     res = map(f, cols)
#     reduce(hcat, res)
# end

# Following a suggestion? Doesn't help.
# @adjoint Base.collect(x) = collect(x), Δ -> (Δ,)


#========== Gradients for TensorCast's functions ==========#

using TensorCast

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

"""
    slicemap(f, A; dims) ≈ mapslices(f, A; dims)

Like `mapcols()`, but for any slice. Gradient is for Zygote only.
"""
function slicemap(f::Function, A::AbstractArray{T,N}, args...; dims) where {T,N}
    code = ntuple(d -> d in dims ? (:) : (*), N)
    B = TensorCast.sliceview(A, code)
    C = [ f(slice, args...) for slice in B ]
    TensorCast.glue(C, code)
end

end # module
