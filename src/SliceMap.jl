
module SliceMap

export mapcols, MapCols, maprows, slicemap, tmapcols, ThreadMapCols

using JuliennedArrays

using Tracker
using Tracker: TrackedReal, TrackedMatrix, track, @grad, data

using ZygoteRules
using ZygoteRules: pullback, @adjoint

#========== Reverse, Eachslice ==========#

"""
    mapcols(f, m) ≈ mapreduce(f, hcat, eachcol(m)) ≈ mapslices(f, m, dims=1)

This is a more efficient version of the functions on the right.
For `f(x::Vector)::Matrix` it reshapes like `mapslices(vec∘f, m, dims=1)`.
For `f(x::Vector)::Number` it skips the reduction, just `reshape(map(f, eachcol(m)),1,:)`.

It provides a gradient for Tracker and Zygote, saving the backward function for each slice.

Any arguments after the matrix are passed to `f` as scalars, i.e.
`mapcols(f, m, args...) = reduce(hcat, f(col, args...) for col in eeachcol(m))`.
They do not get sliced/iterated (unlike `map`), nor are their gradients tracked.

Note that if `f` itself contains parameters, their gradients are also not tracked.
"""
mapcols(f, M, args...) = _mapcols(map, f, M, args...)
tmapcols(f, M, args...) = _mapcols(threadmap, f, M, args...)

function _mapcols(map::Function, f, M::AbstractMatrix, args...)
    res = map(col -> _vec(f(col, args...)), eachcol(M))
    eltype(res) <: AbstractVector ? reduce(hcat, res) : reshape(res,1,:)
end

_vec(x) = x
_vec(A::AbstractArray) = vec(A) # to allow f vector -> matrix, by reshaping

_mapcols(map::Function, f, M::TrackedMatrix, args...) = track(_mapcols, map, f, M, args...)

@grad _mapcols(map::Function, f, M::AbstractMatrix, args...) =
    ∇mapcols(map, map(col -> Tracker.forward(x -> _vec(f(x, args...)), col), eachcol(data(M))), args...)

@adjoint _mapcols(map::Function, f, M::AbstractMatrix, args...) =
    ∇mapcols(map, map(col -> ZygoteRules.pullback(x -> _vec(f(x, args...)), col), eachcol(M)), args)

function ∇mapcols(bigmap, forwards, args...)
    res = map(data∘first, forwards)
    function back(Δ)
        Δcols = eltype(res) <: AbstractVector ? eachcol(data(Δ)) : vec(data(Δ))
        cols = bigmap((fwd, Δcol) -> data(last(fwd)(Δcol)[1]), forwards, Δcols)
        (nothing, nothing, reduce(hcat, cols), map(_->nothing, args)...)
    end
    eltype(res) <: AbstractVector ? reduce(hcat, res) : reshape(res,1,:), back
end

"""
    maprows(f, M) ≈ mapslices(f, M, dims=2)

Like `mapcols()` but for rows.
"""
function maprows(f::Function, M::AbstractMatrix, args...)
    res = map(col -> transpose(_vec(f(col, args...))), eachrow(M))
    eltype(res) <: AbstractArray ? reduce(vcat, res) : reshape(res,:,1)
end

maprows(f::Function, M::TrackedMatrix, args...) = track(maprows, f, M, args...)

@grad maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows(map(row -> Tracker.forward(x -> _vec(f(x, args...)), row), eachrow(data(M))), args)

@adjoint maprows(f::Function, M::AbstractMatrix, args...) =
    ∇maprows(map(row -> ZygoteRules.pullback(x -> _vec(f(x, args...)), row), eachrow(M)), args)

function ∇maprows(forwards, args)
    res = map(transpose∘data∘first, forwards)
    function back(Δ)
        Δrows = eltype(res) <: AbstractArray ? eachrow(data(Δ)) : vec(data(Δ))
        rows = map((fwd, Δrow) -> transpose(data(last(fwd)(Δrow)[1])), forwards, Δrows)
        (nothing, reduce(vcat, rows), map(_->nothing, args)...)
    end
    eltype(res) <: AbstractArray ? reduce(vcat, res) : reshape(res,:,1), back
end

"""
    slicemap(f, A; dims) ≈ mapslices(f, A; dims)

Like `mapcols()`, but for any slice. The function `f` must preserve shape,
e.g. if `dims=(2,4)` then `f` must map matrices to matrices.

The gradient is for Zygote only.

Parameters within the function `f` (if there are any) should be correctly tracked,
which is not the case for `mapcols()`.
"""
function slicemap(f, A::AbstractArray{T,N}, args...; dims) where {T,N}
    code = ntuple(d -> d in dims ? True() : False(), N)
    B = JuliennedArrays.Slices(A, code...)
    C = [ f(slice, args...) for slice in B ]
    JuliennedArrays.Align(C, code...)
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
    d == size(M,1) || error("expected M with $d rows")
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(M))
    B = map(col -> surevec(f(col, args...)), A)
    reduce(hcat, B)
end

surevec(A::AbstractArray) = vec(A)
surevec(x::Number) = SVector(x) # simple way to deal with f vector -> scalar

_MapCols(map::Function, f::Function, M::TrackedMatrix, dval, args...) =
    track(_MapCols, map, f, M, dval, args...)

@grad _MapCols(map::Function, f::Function, M::TrackedMatrix, dval, args...) =
    ∇MapCols(map, f, M, dval, args...)

@adjoint _MapCols(map::Function, f::Function, M::Matrix, dval, args...) =
    ∇MapCols(map, f, M, dval, args...)

function ∇MapCols(bigmap::Function, f::Function, M::AbstractMatrix{T}, dval::Val{d}, args...) where {T,d}
    d == size(M,1) || error("expected M with $d rows")
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

#= JuliennedArrays =#

@adjoint JuliennedArrays.Slices(whole, along...) =
    Slices(whole, along...), Δ -> (Align(Δ, along...), map(_->nothing, along)...)

@adjoint JuliennedArrays.Align(whole, along...) =
    Align(whole, along...), Δ -> (Slices(Δ, along...), map(_->nothing, along)...)

#= Base =#

@adjoint Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector}) =
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)),)


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

@adjoint gluecol(V::AbstractVector) =
    gluecol(V), ΔM -> (collect(eachcol(ΔM)),) # does work!

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

@adjoint function collecteachcol(x)
    collecteachcol(x), dy -> begin
        dx = _zero(x) # _zero is not in ZygoteRules, TODO
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

#========== Threaded Map ==========#

# What KissThreading does is much more complicated, perhaps worth investigating:
# https://github.com/mohamed82008/KissThreading.jl/blob/master/src/KissThreading.jl

# BTW I do the first one because some diffeq maps infer to ::Any,
# else you could use Core.Compiler.return_type(f, Tuple{eltype(x)})

"""
    threadmap(f, A)
    threadmap(f, A, B)

Simple version of `map` using a `Threads.@threads` loop,
or `Threads.@spawn` on Julia >= 1.3.

Only for vectors & really at most two of them, of nonzero length,
with all outputs having the same type.
"""
function threadmap(f::Function, vw::AbstractVector...)
    length(first(vw))==0 && error("can't map over empty vector, sorry")
    length(vw)==2 && (isequal(length.(vw)...) || error("lengths must be equal"))
    out1 = f(first.(vw)...)
    _threadmap(out1, f, vw...)
end
# NB function barrier. Plus two versions:
@static if VERSION >= v"1.3"

    function _threadmap(out1, f, vw...)
        out = Vector{typeof(out1)}(undef, length(first(vw)))
        out[1] = out1
        Threads.@threads for i in 2:length(first(vw))
            @inbounds out[i] = f(getindex.(vw, i)...)
        end
        out
    end

else

    function _threadmap(out1, f, vw...)
        ell = length(first(vw))
        out = Vector{typeof(out1)}(undef, ell)
        out[1] = out1
        Base.@sync for is in Iterators.partition(2:ell, div(ell, Threads.nthreads()))
            Threads.@spawn for i in is
                @inbounds out[i] = f(getindex.(vw, i)...)
            end
        end
        out
    end

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
