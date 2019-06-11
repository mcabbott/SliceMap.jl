# SliceMap.jl

[![Build Status](https://travis-ci.org/mcabbott/SliceMap.jl.svg?branch=master)](https://travis-ci.org/mcabbott/SliceMap.jl)

This package provides some `mapslices`-like functions, 
with gradients for [Flux](https://github.com/FluxML/Flux.jl) and [Zygote](https://github.com/FluxML/Zygote.jl):

```julia
mapcols(f, M) ≈ mapreduce(f, hcat, eachcol(M))
MapCols{d}(f, M)         # where d=size(M,1), for SVector slices
ThreadMapCols{d}(f, M)   # using Threads.@threads

maprows(f, M) ≈ mapslices(f, M, dims=2)

slicemap(f, A; dims) ≈ mapslices(f, A, dims=dims) # only Zygote
```

The capitalised functions differ both in using [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) 
slices, and using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for the gradient of each slice,
instead of the same reverse-mode Tracker/Zygote.
For small slices, this will often be much faster, with or without gradients. 

The package also defines Zygote gradients for the Slice/Align functions in 
[JuliennedArrays](https://github.com/bramtayl/JuliennedArrays.jl), 
and the slice/glue functions in [TensorCast](https://github.com/mcabbott/TensorCast.jl), 
both of which are good ways to roll-your-own `mapslices`-like things.

There are more details & examples at [docs/intro.md](docs/intro.md). 
