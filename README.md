# SliceMap.jl

It would be nice if [Flux](https://github.com/FluxML/Flux.jl) / [Zygote](https://github.com/FluxML/Zygote.jl) worked with `mapslices`, 
or with something generalising that. This package has some quick attempts:

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

# Zygote.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat) # errors
Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]      # Zygote.forward 
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

Of course `mapslices()` does things other than columns of matrices. 
Most of which can be done better with `eachslice()` and `reduce(hcat,...)`, 
maybe with some thought one could just write gradients for those...

Perhaps this is done, at least for Zygote. The views of `eachcol()` have quite inefficient gradients, 
because for each `view()` they make a fresh `zero(A)`, but `collecteachcol()` is efficient:

```julia
@btime Zygote.gradient(m -> sum(sin, mapcols4(fun, m)), $mat1k);  # 45.616 ms, 49.49 MiB
@btime Zygote.gradient(m -> sum(sin, mapcols6(fun, m)), $mat1k);  # 18.655 ms,  3.37 MiB
```

Or for the slice/glue functions in [TensorCast](https://github.com/mcabbott/TensorCast.jl),
which now does some mapslices things (and will soon do many more) by chaining such functions.

```julia
using TensorCast
@cast [i,j] := fun(mat[:,j])[i]                       # same as mapcols

tcm(mat) = @cast out[i,j] := fun(mat[:,j])[i]
Zygote.gradient(m -> sum(sin, tcm(m)), mat)[1]

@btime tcm($mat1k)                                    # 407.176 μs
@btime Zygote.gradient(m -> sum(sin, tcm(m)), $mat1k) # 19.086 ms

ten = rand(1:9, 3,10,2)
@cast zed[i,j,k] := fun(ten[i,:,k])[j]
Zygote.gradient(m -> sum(sin, @cast zed[i,j,k] := fun(m[i,:,k])[j]  nolazy), ten)[1]
```

The function `slicemap(f, A, dims)` uses these slice/glue functions, 
without having to write index notation. 

Issues about mapslices:
* https://github.com/FluxML/Zygote.jl/issues/92
* https://github.com/FluxML/Flux.jl/issues/741

Other packages which define gradients of possible interest:
* https://github.com/GiggleLiu/LinalgBackwards.jl
* https://github.com/mcabbott/ArrayAllez.jl

I added some tests: 
[![Build Status](https://travis-ci.org/mcabbott/SliceMap.jl.svg?branch=master)](https://travis-ci.org/mcabbott/SliceMap.jl)

<!--
AD packages this could perhaps support, quite the zoo:
* https://github.com/invenia/Nabla.jl
* https://github.com/dfdx/Yota.jl
* https://github.com/denizyuret/AutoGrad.jl
* https://github.com/Roger-luo/YAAD.jl
* And perhaps one day, just https://github.com/JuliaDiff/ChainRules.jl
-->
