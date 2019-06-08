# SliceMap.jl

[![Build Status](https://travis-ci.org/mcabbott/SliceMap.jl.svg?branch=master)](https://travis-ci.org/mcabbott/SliceMap.jl)

This package provides some `mapslices`-like functions, 
with gradients for [Flux](https://github.com/FluxML/Flux.jl) and [Zygote](https://github.com/FluxML/Zygote.jl):

```julia
mapcols(f, M) ≈ mapreduce(f, hcat, eachcol(M))
MapCols{d}(f, M) # where d=size(M,1), for StaticArrays

maprows(f, M) ≈ mapreduce(f, vcat, eachrow(M))

slicemap(f, A; dims) ≈ mapslices(f, A, dims)
```

### An example

```julia
mat = rand(1:9, 3,10)
fun(x) = 2 .+ x.^2
mapslices(fun, mat, dims=1)

using SliceMap
mapcols(fun, mat)     # eachcol(m)
MapCols{3}(fun, mat)  # reinterpret(SArray,...)

using Tracker, Zygote, ForwardDiff
ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat)

Tracker.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]     # Tracker.forward per slice
Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]  # ForwardDiff on slices

Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]      # Zygote.forward per slice
Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]
```

These are a bit faster than `mapslices` too. Although storing all the backward functions, 
which is what `mapcols` does, seems not to be so quick:

```julia
using BenchmarkTools
mat1k = rand(3,1000);

@btime mapreduce(fun, hcat, eachcol($mat1k)) # 1.522 ms
@btime mapslices(fun, $mat1k, dims=1)        # 1.017 ms

@btime mapcols(fun, $mat1k)                  #   399.016 μs
@btime MapCols{3}(fun, $mat1k)               #    15.564 μs
@btime MapCols(fun, $mat1k)                  #    16.774 μs  without size

@btime ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), $mat1k); # 372.705 ms
@btime Tracker.gradient(m -> sum(sin, mapcols(fun, m)), $mat1k);               #  70.203 ms
@btime Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), $mat1k);            #     146.561 μs, 330.51 KiB
@btime Zygote.gradient(m -> sum(sin, mapcols(fun, m)), $mat1k);                #  20.018 ms, 3.82 MiB
@btime Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), $mat1k);             #     245.550 μs
```

It also provides Zygote gradients for the slice/glue functions in 
[TensorCast](https://github.com/mcabbott/TensorCast.jl),
which can be used to write many mapslices-like operations.
(The function `slicemap(f, A, dims)` uses these functions, without having to write index notation.)

```julia
using TensorCast
@cast [i,j] := fun(mat[:,j])[i]                       # same as mapcols

tcm(mat) = @cast out[i,j] := fun(mat[:,j])[i]
Zygote.gradient(m -> sum(sin, tcm(m)), mat)[1]

@btime tcm($mat1k)                                    # 407.176 μs
@btime Zygote.gradient(m -> sum(sin, tcm(m)), $mat1k) # 19.086 ms
```

### Elsewhere

Issues about mapslices:
* https://github.com/FluxML/Zygote.jl/issues/92
* https://github.com/FluxML/Flux.jl/issues/741

Other packages which define gradients of possible interest:
* https://github.com/GiggleLiu/LinalgBackwards.jl
* https://github.com/mcabbott/ArrayAllez.jl

AD packages this could perhaps support, quite the zoo:
* https://github.com/invenia/Nabla.jl
* https://github.com/dfdx/Yota.jl
* https://github.com/denizyuret/AutoGrad.jl
* https://github.com/Roger-luo/YAAD.jl
* And perhaps one day, just https://github.com/JuliaDiff/ChainRules.jl
