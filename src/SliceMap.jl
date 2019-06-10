
module SliceMap

export mapcols, MapCols, maprows, slicemap, ThreadMapCols

using MacroTools, Requires, WeightedArrays, TensorCast, JuliennedArrays

using Tracker
using Tracker: TrackedMatrix, track, @grad, data

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
    reduce(vcat, [ transpose(surevec(f(col, args...))) for col in eachrow(M) ])

maprows(f::Function, M::TrackedMatrix, args...) = track(maprows, f, M, args...)

@grad maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows([ Tracker.forward(x -> surevec(f(x, args...)), row) for row in eachrow(data(M)) ], args)

function ∇maprows(forwards, args)
    reduce(vcat, map(transpose∘data∘first, forwards)), Δ -> begin
        rows = [ data(last(fwd)(Δrow)[1]) for (fwd, Δrow) in zip(forwards, eachrow(data(Δ))) ]
        (nothing, reduce(vcat, transpose.(rows)), map(_->nothing, args)...)
    end
end

"""
    slicemap(f, A; dims) ≈ mapslices(f, A; dims)

Like `mapcols()`, but for any slice. The function `f` must preserve shape,
e.g. `dims=(2,4)` then `f` must map matrices to matrices.

The gradient is for Zygote only.
"""
function slicemap(f::Function, A::AbstractArray{T,N}, args...; dims) where {T,N}
    code = ntuple(d -> d in dims ? (:) : (*), N)
    B = TensorCast.sliceview(A, code)
    C = [ f(slice, args...) for slice in B ]
    TensorCast.glue(C, code)
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

MapCols{d}(f::Function, M::AbstractMatrix, args...) where {d} =
    _MapCols(f, M, Val(d), Val(false), args...)

function _MapCols(f::Function, M::Matrix{T}, ::Val{d}, tval::Val, args...) where {T,d}
    d == size(M,1) || error("expected M with $d columns")
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(M))
    B = maybethreadmap(col -> surevec(f(col, args...)), A, tval)
    reduce(hcat, B)
end

_MapCols(f::Function, M::TrackedMatrix, dval, tval, args...) =
    track(_MapCols, f, M, dval, tval, args...)

@grad _MapCols(f::Function, M::TrackedMatrix, dval, tval, args...) =
    ∇MapCols(f, M, dval, tval, args...)

function ∇MapCols(f::Function, M::AbstractMatrix{T}, dval::Val{d}, tval::Val, args...) where {T,d}

    d == size(M,1) || error("expected M with $d columns")
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(data(M)))

    dualcol = SVector(ntuple(j->ForwardDiff.Dual(0, ntuple(i->i==j ? 1 : 0, dval)...), dval))
    C = maybethreadmap(col -> surevec(f(col + dualcol, args...)), A, tval)

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
        (nothing, ∇M, nothing, nothing, map(_->nothing, args)...)
    end
    Z, back
end

#========== Gradients for Zygote ==========#

# @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
# end

@init @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("zygote.jl")

#========== Experimenting with gradients for for eachslice / reduce ==========#

export gluecol, collecteachcol
export mapcols2, mapcols4, mapcols5, mapcols6, mapcols7

gluecol(V::AbstractVector{<:AbstractVector}) = reduce(hcat, V)

#=
gluecol(V::AbstractVector{<:TrackedVector}) = track(gluecol, V)

@grad function gluecol(V::AbstractVector)
    gluecol(data.(V)), ΔM -> (collect(eachcol(data(ΔM))),) # doesn't work
end
=#

function mapcols2(f, A)
    cols = [A[:,c] for c=1:size(A,2)]
    res = f.(cols)
    gluecol(res)
end

# Apply that straight to reduce(hcat,...)

function mapcols4(f, A)
    cols = [view(A,:,c) for c=1:size(A,2)]
    res = map(f, cols)
    reduce(hcat, res)
end

# Surprisingly dy for eachcol seems to know the answer?
# typeof(dy) = NamedTuple{(:f, :iter),Tuple{NamedTuple{(:A,),Tuple{Array{Float64,2}}},Array{Nothing,1}}}
# dy = (f = (A = [47.9325 51.3781
# Which means this works... but uses as much memory as gradient of array of views:

#=@adjoint function eachcol(x::AbstractMatrix)
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

#========== Threaded Map ==========#

# What KissThreading does is much more complicated, perhaps worth investigating:
# https://github.com/mohamed82008/KissThreading.jl/blob/master/src/KissThreading.jl

function threadmap(f::Function, v::AbstractVector)
    length(v)==0 && error("can't map over empty vector, sorry")
    out1 = f(first(v))
    _threadmap(out1, f, v)
end
# NB barrier
function _threadmap(out1, f, v)
    out = Vector{typeof(out1)}(undef, length(v))
    out[1] = out1
    Threads.@threads for i=2:length(v)
        @inbounds out[i] = f(v[i])
    end
    out
end

# This switch is fast inside ∇MapCols, after many attempts!
maybethreadmap(f, v, ::Val{true}) = threadmap(f, v)
maybethreadmap(f, v, ::Val{false}) = map(f, v)

struct ThreadMapCols{d} end

"""
    ThreadMapCols{d}(f, m::Matrix, args...)

Like `MapCols` but with multi-threading!
"""
ThreadMapCols(f::Function, M::AT, args...) where {AT<:WeightedArrays.MaybeWeightedMatrix} =
    ThreadMapCols{size(M,1)}(f, M, args...)

ThreadMapCols{d}(f::Function, M::WeightedMatrix, args...) where {d} =
    Weighted(ThreadMapCols{d}(f, M.array, args...), M.weights, M.opt)

ThreadMapCols{d}(f::Function, M::AbstractMatrix, args...) where {d} =
    _MapCols(f, M, Val(d), Val(true), args...)


end # module
