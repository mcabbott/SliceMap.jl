
module SliceMap

export MapCols, mapcols

#========== Reverse, Eachslice ==========#

"""
    mapcols(f, m::Matrix, args...) = reduce(hcat, f(c, args...) for c in eachcol(M))

When `m::TrackedMatrix`, it saves the backward function for each slice.
"""
mapcols(f::Function, M::Matrix, args...) =
    reduce(hcat, [ rvec(f(col, args...)) for col in eachcol(M) ])

using Tracker
using Tracker: TrackedMatrix, track, @grad, data

mapcols(f::Function, M::TrackedMatrix, args...) = track(mapcols, f, M, args...)

@grad function mapcols(f::Function, M::TrackedMatrix, args...)
    res = [ Tracker.forward(x -> rvec(f(x, args...)), col) for col in eachcol(data(M)) ]
    fwd = reduce(hcat, data.(first.(res)))
    function back(Δ)
        cols = [ data((last(res[c]))(Δcol)[1]) for (c, Δcol) in enumerate(eachcol(data(Δ))) ]
        ∇M = reduce(hcat, cols)
        (nothing, ∇M, map(_->nothing, args)...)
    end
    fwd, back
end

using Zygote
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

end # module
