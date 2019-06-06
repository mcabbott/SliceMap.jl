
module SliceMap

export MapCols, mapcols, maprows


#========== Gradient Macro ==========#

using MacroTools, Tracker, Zygote
using Tracker: TrackedMatrix, track, @grad, data
using Zygote: @adjoint, _zero

macro gradadjoint(ex)
    quote
        # $(Zygote.gradm(ex)) # this doesn't work
        $(trackergrad(ex))
    end
end

# Copied from https://github.com/FluxML/Tracker.jl/blob/master/src/Tracker.jl#L55
function trackergrad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  isexpr(name, :(::)) || (name = :(::typeof($name)))
  insert!(args, 1+isexpr(args[1], :parameters) , name)
  MacroTools.@q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end


#========== Reverse, Eachslice ==========#

using WeightedArrays

"""
    mapcols(f, m::Matrix, args...) = reduce(hcat, f(c, args...) for c in eachcol(M))

When `m::TrackedMatrix`, it saves the backward function for each slice.
All further arguments are scalar constants, i.e. they do not get sliced/iterated (unlike `map`)
nor are their gradients tracked.
"""
mapcols(f::Function, M::AbstractMatrix, args...) =
    reduce(hcat, [ surevec(f(col, args...)) for col in eachcol(M) ])

mapcols(f::Function, M::WeightedMatrix, args...) =
    Weighted(mapcols(f, M.array, args...), M.weights, M.opt)

surevec(x::Number) = [x] # to allow f vector -> scalar, as mapslices does
surevec(A) = vec(A)      # to allow f vector -> matrix, by reshaping

mapcols(f::Function, M::TrackedMatrix, args...) = track(mapcols, f, M, args...)

@grad function mapcols(f::Function, M::AbstractMatrix, args...)
    res = [ Tracker.forward(x -> surevec(f(x, args...)), col) for col in eachcol(data(M)) ]
    fwd = reduce(hcat, data.(first.(res)))
    function back(Δ)
        cols = [ data((last(res[c]))(Δcol)[1]) for (c, Δcol) in enumerate(eachcol(data(Δ))) ]
        ∇M = reduce(hcat, cols)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end

@adjoint function mapcols(f::Function, M::Matrix, args...)
    res = [ Zygote.forward(x -> surevec(f(x, args...)), col) for col in eachcol(M) ]
    fwd = reduce(hcat, first.(res))
    function back(Δ)
        cols = [ (last(res[c]))(Δcol)[1] for (c, Δcol) in enumerate(eachcol(Δ)) ]
        ∇M = reduce(hcat, cols)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end

maprows(f::Function, M::AbstractMatrix, args...) =
    reduce(vcat, [ surerow(f(col, args...)) for col in eachrow(M) ])

surerow(x) = transpose(surevec(x))

maprows(f::Function, M::TrackedMatrix, args...) = track(maprows, f, M, args...)

@grad function maprows(f::Function, M::AbstractMatrix, args...)
    res = [ Tracker.forward(x -> surerow(f(x, args...)), row) for row in eachrow(data(M)) ]
    fwd = reduce(vcat, data.(first.(res)))
    function back(Δ)
        rows = [ data((last(res[r]))(Δrow)[1]) for (r, Δrow) in enumerate(eachrow(data(Δ))) ]
        ∇M = reduce(vcat, rows)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end


#========== Forward, Static ==========#

using StaticArrays, ForwardDiff, WeightedArrays

struct MapCols{d} end

"""
    MapCols{d}(f, m::Matrix, args...)

Expects `f(::SVector{d}, args...)` and maps this over the columns, `d = size(M,1)`.
Doesn't expect `f` to return a staticarray, just an array.

When `m::TrackedMatrix`, it uses `ForwardDiff` to calculate the gradient of each slice.
The second point of keeping one type parameter is that the dual numbers needed depend on this.

    MapCols{d}(f, m::Weighted, args...)
Takes `m.weights` along for the ride.
"""
MapCols(f::Function, M::WeightedArrays.MaybeWeightedMatrix, args...) =
    MapCols{size(M,1)}(f, M, args...)

MapCols{d}(f::Function, M::WeightedMatrix, args...) where {d} =
    Weighted(MapCols{d}(f, M.array, args...), M.weights, M.opt)

MapCols{d}(f::Function, M::AbstractMatrix, args...) where {d} = _MapCols(f, M, Val(d), args...)

function _MapCols(f::Function, M::Matrix{T}, ::Val{d}, args...) where {T,d}
    d == size(M,1) || error("expected M with $d columns")
    # @cast A[c]{r:d} := M[r,c] assert
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(M))
    B = map(col -> surevec(f(col, args...)), A)
    reduce(hcat, B)
    # maybestaticgluecols(B)
end

# surevec(x::MArray) = Array(x) # avoid making a huge MArray, ad

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

# surevecS(x::Number) = @SVector [x]
# surevecS(A) = vec(A) # like surevec

_MapCols(f::Function, M::TrackedMatrix, dval, args...) = track(_MapCols, f, M, dval, args...)

@grad _MapCols(f::Function, M::TrackedMatrix, dval, args...) = ∇MapCols(f, M, dval, args...)

@adjoint _MapCols(f::Function, M::Matrix, dval, args...) = ∇MapCols(f, M, dval, args...)

function ∇MapCols(f::Function, M::AbstractMatrix{T}, dval::Val{d}, args...) where {T,d}

    d == size(M,1) || error("expected M with $d columns")
    # @cast A[c]{r:d} := data(M)[r,c]
    A = reinterpret(SArray{Tuple{d}, T, 1, d}, vec(data(M)))

    dualcol = SVector(ntuple(j->ForwardDiff.Dual(0, ntuple(i->i==j ? 1 : 0, dval)...), dval))

    # C = [ surevec(f(col .+ dualcol, args...)) for col in A ]
    C = map(col -> surevec(f(col .+ dualcol, args...)), A)

    # Z = reduce(hcat, [ ForwardDiff.value.(full) for full in C ])
    Z = reduce(hcat, map(col -> ForwardDiff.value.(col), C))

    function back(ΔZ)
        # accum = zero(eltype(data(ΔZ)))
        # ∇M = similar(data(M)) .+ zero(first(data(ΔZ)))
        ∇M = zeros(eltype(data(ΔZ)), size(M))
        @inbounds for c=1:size(M,2)
            part = ForwardDiff.partials.(C[c])
            for r=1:d
                # ∇M[r,c] = 0
                # accum = 0
                for i=1:size(ΔZ,1)
                    ∇M[r,c] += data(ΔZ)[i,c] * part[i].values[r]
                    # parti = ForwardDiff.partials(C[c][i])
                    # ∇M[r,c] += data(ΔZ)[i,c] * parti.values[r]
                    # accum += data(ΔZ)[i,c] * part[i].values[r]
                end
                # ∇M[r,c] = accum
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


end # module
