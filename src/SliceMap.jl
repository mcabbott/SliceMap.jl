
module SliceMap

export mapcols, MapCols, maprows, slicemap, tmapcols, ThreadMapCols

using MacroTools, Requires, TensorCast, JuliennedArrays

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
mapcols(f::Function, M, args...) = _mapcols(map, f, M, args...)
tmapcols(f::Function, M, args...) = _mapcols(threadmap, f, M, args...)

_mapcols(map::Function, f::Function, M::AbstractMatrix, args...) =
    reduce(hcat, map(col -> surevec(f(col, args...)), eachcol(M)))

surevec(x::Number) = [x] # to allow f vector -> scalar, as mapslices does
surevec(A) = vec(A)      # to allow f vector -> matrix, by reshaping

_mapcols(map::Function, f::Function, M::TrackedMatrix, args...) = track(_mapcols, map, f, M, args...)

@grad _mapcols(map::Function, f::Function, M::AbstractMatrix, args...) =
    ∇mapcols(map, map(col -> Tracker.forward(x -> surevec(f(x, args...)), col), eachcol(data(M))), args...)

function ∇mapcols(bigmap, forwards, args...)
    reduce(hcat, map(data∘first, forwards)), Δ -> begin
        cols = bigmap((fwd, Δcol) -> data(last(fwd)(Δcol)[1]), forwards, eachcol(data(Δ)))
        (nothing, nothing, reduce(hcat, cols), map(_->nothing, args)...)
    end
end

"""
    maprows(f, M) ≈ mapslices(f, M, dims=2)

Like `mapcols()` but for rows.
"""
maprows(f::Function, M::AbstractMatrix, args...) =
    reduce(vcat, map(col -> transpose(surevec(f(col, args...))), eachrow(M)))

maprows(f::Function, M::TrackedMatrix, args...) = track(maprows, f, M, args...)

@grad maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows(map(row -> Tracker.forward(x -> surevec(f(x, args...)), row), eachrow(data(M))), args)

function ∇maprows(forwards, args)
    reduce(vcat, map(transpose∘data∘first, forwards)), Δ -> begin
        rows = map((fwd, Δrow) -> data(last(fwd)(Δrow)[1]), forwards, eachrow(data(Δ)))
        (nothing, reduce(vcat, transpose.(rows)), map(_->nothing, args)...)
    end
end

"""
    slicemap(f, A; dims) ≈ mapslices(f, A; dims)

Like `mapcols()`, but for any slice. The function `f` must preserve shape,
e.g. if `dims=(2,4)` then `f` must map matrices to matrices.

The gradient is for Zygote only.
"""
function slicemap(f::Function, A::AbstractArray{T,N}, args...; dims) where {T,N}
    code = ntuple(d -> d in dims ? (:) : (*), N)
    B = TensorCast.sliceview(A, code)
    C = [ f(slice, args...) for slice in B ]
    TensorCast.glue(C, code)
end

#========== Forward, Static ==========#

using StaticArrays, ForwardDiff

struct MapCols{d} end

"""
    MapCols{d}(f, m::Matrix, args...)

Similar to `mapcols(f, m, args...)`, but slices `m` into `SVector{d}` columns.
Their length `d = size(M,1)` should ideally be provided for type-stability, but is not required.

The gradient for Tracker and Zygote uses `ForwardDiff` on each slice.
"""
MapCols(f::Function, M::AbstractMatrix, args...) =
    MapCols{size(M,1)}(f, M, args...)

MapCols{d}(f::Function, M::AbstractMatrix, args...) where {d} =
    _MapCols(map, f, M, Val(d), args...)

function _MapCols(map::Function, f::Function, M::Matrix{T}, ::Val{d}, args...) where {T,d}
    d == size(M,1) || error("expected M with $d columns")
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(M))
    B = map(col -> surevec(f(col, args...)), A)
    reduce(hcat, B)
end

_MapCols(map::Function, f::Function, M::TrackedMatrix, dval, args...) =
    track(_MapCols, map, f, M, dval, args...)

@grad _MapCols(map::Function, f::Function, M::TrackedMatrix, dval, args...) =
    ∇MapCols(map, f, M, dval, args...)

function ∇MapCols(bigmap::Function, f::Function, M::AbstractMatrix{T}, dval::Val{d}, args...) where {T,d}
    d == size(M,1) || error("expected M with $d columns")
    k = size(M,2)

    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(data(M)))

    dualcol = SVector(ntuple(j->ForwardDiff.Dual(0, ntuple(i->i==j ? 1 : 0, dval)...), dval))
    C = bigmap(col -> surevec(f(col + dualcol, args...)), A)

    Z = reduce(hcat, map(col -> ForwardDiff.value.(col), C))

    function back(ΔZ)
        S = promote_type(T, eltype(data(ΔZ)))
        ∇M = zeros(S, size(M))
        @inbounds for c=1:k
            part = ForwardDiff.partials.(C[c])
            for r=1:d
                for i=1:size(ΔZ,1)
                    ∇M[r,c] += data(ΔZ)[i,c] * part[i].values[r]
                end
            end
        end
        (nothing, nothing, ∇M, nothing, map(_->nothing, args)...)
    end
    Z, back
end

#========== Gradients for Zygote ==========#

# @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

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

# BTW I do the first one because some diffeq maps infer to ::Any,
# else you could use Core.Compiler.return_type(f, Tuple{eltype(x)})

"""
    threadmap(f, A)
    threadmap(f, A, B)

Simple version of `map` using a `Threads.@threads` loop;
only for vectors & really at most two of them, of nonzero length,
with all outputs having the same type.
"""
function threadmap(f::Function, vw::AbstractVector...)
    length(first(vw))==0 && error("can't map over empty vector, sorry")
    length(vw)==2 && (isequal(length.(vw)...) || error("lengths must be equal"))
    out1 = f(first.(vw)...)
    _threadmap(out1, f, vw...)
end
# NB barrier
function _threadmap(out1, f, vw...)
    out = Vector{typeof(out1)}(undef, length(first(vw)))
    out[1] = out1
    Threads.@threads for i=2:length(first(vw))
        @inbounds out[i] = f(getindex.(vw, i)...)
    end
    out
end

# Collect generators to allow indexing
threadmap(f::Function, v) = threadmap(f, collect(v))
threadmap(f::Function, v, w) = threadmap(f, collect(v), collect(w))
threadmap(f::Function, v, w::AbstractVector) = threadmap(f, collect(v), w)
threadmap(f::Function, v::AbstractVector, w) = threadmap(f, v, collect(w))

struct ThreadMapCols{d} end

"""
    ThreadMapCols{d}(f, m::Matrix, args...)

Like `MapCols` but with multi-threading!
"""
ThreadMapCols(f::Function, M::AbstractMatrix, args...) =
    ThreadMapCols{size(M,1)}(f, M, args...)

ThreadMapCols{d}(f::Function, M::AbstractMatrix, args...) where {d} =
    _MapCols(threadmap, f, M, Val(d), args...)


end # module
