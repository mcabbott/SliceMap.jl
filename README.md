# SliceMap.jl

It would be nice if [Flux](https://github.com/FluxML/Flux.jl) worked with `mapslices`, 
or with something generalising that. This package has some quick attempts:

```julia
mat = rand(1:99, 3,10)
fun(x) = 2 .+ x.^2
mapslices(fun, mat, dims=1)

using SliceMap

mapcols(fun, mat)     # eachcol(m)
MapCols{3}(fun, mat)  # reinterpret(SArray,...)

using Tracker, Zygote, ForwardDiff
ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat)

Tracker.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]     # Tracker.forward per slice
Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]  # ForwardDiff on slices

# Zygote.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat)
Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]      # Zygote.forward 
Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]
```

These are a bit faster than `mapslices` too:

```julia
mat1k = rand(3,1000);

@btime mapslices(fun, $mat1k, dims=1)  # 1.017 ms
@btime mapcols(fun, $mat1k)            #   399.016 μs
@btime MapCols{3}(fun, $mat1k)         #    46.733 μs
@btime MapCols(fun, $mat1k)            #    59.471 μs without size

@btime ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), $mat1k); # 372.705 ms
@btime Tracker.gradient(m -> sum(sin, mapcols(fun, m)), $mat1k);               #  70.203 ms
@btime Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), $mat1k);            #     255.032 μs
@btime Zygote.gradient(m -> sum(sin, mapcols(fun, m)), $mat1k);                #  20.018 ms
@btime Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), $mat1k);             #     354.112 μs
```

Of course `mapslices()` does things other than columns of matrices. 
Most of which can be done better with `eachslice()` and `reduce(hcat,...)`, 
maybe with some thought one could just write gradients for those. 

Or for the slice/glue functions in [TensorCast](https://github.com/mcabbott/TensorCast.jl),
which now does some mapslices things (and will soon do many more) by chaining such functions.

