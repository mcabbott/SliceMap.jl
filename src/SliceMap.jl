
module SliceMap

export MapCols, mapcols

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
    reduce(hcat, [ rvec(f(col, args...)) for col in eachcol(M) ])

mapcols(f::Function, M::WeightedMatrix, args...) =
    Weighted(mapcols(f, M.array, args...), M.weights, M.opt)

mapcols(f::Function, M::TrackedMatrix, args...) = track(mapcols, f, M, args...)

@gradadjoint function mapcols(f::Function, M::AbstractMatrix, args...)
    res = [ Tracker.forward(x -> rvec(f(x, args...)), col) for col in eachcol(data(M)) ]
    fwd = reduce(hcat, data.(first.(res)))
    function back(Δ)
        cols = [ data((last(res[c]))(Δcol)[1]) for (c, Δcol) in enumerate(eachcol(data(Δ))) ]
        ∇M = reduce(hcat, cols)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end

# @gradadjoint not yet working
Zygote.@adjoint function mapcols(f::Function, M::Matrix, args...)
    res = [ Zygote.forward(x -> rvec(f(x, args...)), col) for col in eachcol(data(M)) ]
    fwd = reduce(hcat, data.(first.(res)))
    function back(Δ)
        cols = [ data((last(res[c]))(Δcol)[1]) for (c, Δcol) in enumerate(eachcol(data(Δ))) ]
        ∇M = reduce(hcat, cols)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end

maprows(f::Function, M::AbstractMatrix, args...) =
    reduce(vcat, [ tvec(f(col, args...)) for col in eachrow(M) ])

maprows(f::Function, M::TrackedMatrix, args...) = track(maprows, f, M, args...)

@gradadjoint function maprows(f::Function, M::AbstractMatrix, args...)
    res = [ Tracker.forward(x -> tvec(f(x, args...)), row) for row in eachrow(data(M)) ]
    fwd = reduce(vcat, data.(first.(res)))
    function back(Δ)
        rows = [ data((last(res[r]))(Δrow)[1]) for (r, Δrow) in enumerate(eachrow(data(Δ))) ]
        ∇M = reduce(vcat, rows)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end


#========== Forward, Static ==========#

using TensorCast, StaticArrays, WeightedArrays

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

MapCols{d}(f::Function, M::WeightedMatrix, args...)  where {d} =
    Weighted(MapCols{d}(f, M.array, args...), M.weights, M.opt)

function MapCols{d}(f::Function, M::Matrix, args...) where {d}
    @cast A[c]{r:d} := M[r,c] assert
    reduce(hcat, [ rvec(f(acol, args...)) for acol in A ])

    # TODO: call some function which static-glues if possible...
    # TensorCast.auto_glue(map(col -> rvec(f(col, args...)), A), (:,*))

    # TODO: can I thread this? Is it even safe to do so?
    # https://github.com/mohamed82008/KissThreading.jl
end

rvec(x::Number) = [x] # to allow for f vector -> scalar, as mapslices does
rvec(x::StaticArray) = vec(Array(x)) # to avoid creating a giant staticarray, as reduce(hcat would otherwise do
rvec(A) = vec(A) # LinearAlgebra.

tvec(x) = transpose(rvec(x))

using ForwardDiff

MapCols{d}(f::Function, M::TrackedMatrix, args...) where {d} = track(MapCols, f, M, Val(d), args...)

@grad function MapCols(f::Function, M::TrackedMatrix, dval::Val{d}, args...) where {d}

    @cast A[c]{r:d} := M.data[r,c]
    dualcol = SVector(ntuple(j->ForwardDiff.Dual(0, ntuple(i->i==j ? 1 : 0, dval)...), dval))

    C = [ rvec(f(acol .+ dualcol, args...)) for acol in A ]

    Z = reduce(hcat, [ ForwardDiff.value.(full) for full in C ]) # full is not an SVector here

    function back(ΔZ)
        ∇M = similar(data(M)) .+ zero(first(data(ΔZ)))
        @inbounds for c=1:size(M,2)
            part = ForwardDiff.partials.(C[c])
            for r=1:d
                ∇M[r,c] = 0
                for i=1:size(ΔZ,1)
                    ∇M[r,c] += data(ΔZ)[i,c] * part[i].values[r]
                end
            end
        end
        (nothing, ∇M, nothing, map(_->nothing, args)...)
    end

    Z, back
end

# TODO make a _MapCols which always takes Val(d), then unite these

Zygote.@adjoint function MapCols{d}(f::Function, M::Matrix, args...) where {d} # no dval!

    @cast A[c]{r:d} := M[r,c]
    dualcol = SVector(ntuple(j->ForwardDiff.Dual(0, ntuple(i->i==j ? 1 : 0, Val(d))...), Val(d)))

    C = [ rvec(f(acol .+ dualcol, args...)) for acol in A ]

    Z = reduce(hcat, [ ForwardDiff.value.(full) for full in C ])

    function back(ΔZ)
        ∇M = similar(data(M)) .+ zero(first(data(ΔZ)))
        @inbounds for c=1:size(M,2)
            part = ForwardDiff.partials.(C[c])
            for r=1:d
                ∇M[r,c] = 0
                for i=1:size(ΔZ,1)
                    ∇M[r,c] += data(ΔZ)[i,c] * part[i].values[r]
                end
            end
        end
        (nothing, ∇M, map(_->nothing, args)...) # changed!
    end

    Z, back
end

#========== Gradient for eachslice / reduce ==========#

export gluecol, mapcols2, mapcols4, mapcols5, mapcols6, mapcols7

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

Zygote.@adjoint function eachcol(x::AbstractMatrix)
    eachcol(x), dy -> (dy.f.A,) #= begin
        @show typeof(dy) dy
        dx = zero(x) .+ 0.0  # zeros(eltype(dy), size(x))
        foreach(copyto!, eachcol(dx), dy)
        (dx,)
    end =#
end

# @adjoint eachcol(x) = eachcol(x), dy -> (dy.f.A,)

function mapcols5(f, A)
    cols = collect(eachcol(A))
    res = map(f, cols)
    reduce(hcat, res)
end

collecteachcol(x) = collect(eachcol(x))

Zygote.@adjoint function collecteachcol(x)
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

end # module
