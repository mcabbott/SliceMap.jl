# SliceMap.jl

Some examples & benchmarks. Times shown were on Julia 1.2 I think, 
and some have improved quite a bit since. 

## Simple example

```julia
mat = rand(1:9, 3,10)
fun(x) = 2 .+ x.^2
mapslices(fun, mat, dims=1)

using SliceMap
mapcols(fun, mat)     # eachcol(m)
MapCols{3}(fun, mat)  # reinterpret(SArray,...)

using ForwardDiff, Tracker, Zygote
ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat)

Tracker.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]     # Tracker.forward per slice
Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]  # ForwardDiff on slices

Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]      # Zygote.forward per slice
Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]
```

These are a bit faster than `mapslices` too. Although storing all the backward functions, 
which is what `mapcols` does, has some overhead:

```julia
using BenchmarkTools
mat1k = rand(3,1000);

@btime mapreduce(fun, hcat, eachcol($mat1k)) # 1.522 ms,  11.80 MiB
@btime mapslices(fun, $mat1k, dims=1)        # 1.017 ms,     329.92 KiB

@btime mapcols(fun, $mat1k)                  #   399.016 μs, 219.02 KiB
@btime MapCols{3}(fun, $mat1k)               #    15.564 μs,  47.16 KiB
@btime MapCols(fun, $mat1k)                  #    16.774 μs (without slice size)

@btime ForwardDiff.gradient(m -> sum(mapslices(fun, m, dims=1)), $mat1k); # 329.305 ms
@btime Tracker.gradient(m -> sum(mapcols(fun, m)), $mat1k);               #  70.203 ms
@btime Tracker.gradient(m -> sum(MapCols{3}(fun, m)), $mat1k);            #      51.129 μs, 282.92 KiB
@btime Zygote.gradient(m -> sum(mapcols(fun, m)), $mat1k);                #  20.454 ms,   3.52 MiB
@btime Zygote.gradient(m -> sum(MapCols{3}(fun, m)), $mat1k);             #      28.229 μs, 164.63 KiB
```

On recent versions of Julia, `mapcols` has become much faster, 5-10 times. 

FWIW, times on Julia 1.7-dev & and M1 mac, April 2021:

```
@btime mapreduce(fun, hcat, eachcol($mat1k)) # 616.250 μs (2317 allocations: 11.69 MiB)
@btime mapslices(fun, $mat1k, dims=1)        # 230.250 μs (7499 allocations: 298.23 KiB)
@btime mapcols(fun, $mat1k)                  #  26.750 μs (1003 allocations: 140.83 KiB)
@btime MapCols{3}(fun, $mat1k)               #   6.067 μs (9 allocations: 47.20 KiB)
@btime MapCols(fun, $mat1k)                  #   6.217 μs (9 allocations: 47.20 KiB)

@btime ForwardDiff.gradient(m -> sum(mapslices(fun, m, dims=1)), $mat1k); # 70.815 ms (1877210 allocations: 210.64 MiB)
@btime Tracker.gradient(m -> sum(mapcols(fun, m)), $mat1k);               # 29.840 ms (598046 allocations: 26.20 MiB)
@btime Tracker.gradient(m -> sum(MapCols{3}(fun, m)), $mat1k);            # 25.833 μs (60 allocations: 283.23 KiB)
```

## Other packages

This package also provides Zygote gradients for the Slice/Align functions in 
[JuliennedArrays](https://github.com/bramtayl/JuliennedArrays.jl),
which can be used to write many mapslices-like operations:

```julia
using JuliennedArrays
jumap(f,m) = Align(map(f, Slices(m, True(), False())), True(), False())
jumap1(f,m) = Align(map(f, Slices(m, 1)), 1)
jumap(fun, mat)                                          # same as mapcols
jumap1(fun, mat)
Zygote.gradient(m -> sum(sin, jumap(fun, m)), mat)[1]

@btime jumap(fun, $mat1k);                               #     44.823 μs
@btime jumap1(fun, $mat1k);                              #     11.805 μs, really?
@btime Zygote.gradient(m -> sum(jumap(fun, m)), $mat1k); # 26.110 ms
@btime Zygote.gradient(m -> sum(jumap1(fun, m)), $mat1k) #    412.904 μs, really?
```

Similar gradients also moved to [TensorCast](https://github.com/mcabbott/TensorCast.jl):

```julia
using TensorCast
@cast [i,j] := fun(mat[:,j])[i]                   # same as mapcols

tcm(mat) = @cast out[i,j] := fun(mat[:,j])[i]
Zygote.gradient(m -> sum(sin, tcm(m)), mat)[1]

@btime tcm($mat1k)                                #    427.907 μs
@btime Zygote.gradient(m -> sum(tcm(m)), $mat1k); # 18.358 ms
```

The original purpose of `MapCols`, with ForwardDiff on slices, was that this works well when
the function being mapped integrates some differential equation. 

```julia
using DifferentialEquations, ParameterizedFunctions
ode = @ode_def begin
  du = ( - k2 * u )/(k1 + u) # an equation with 2 parameters
end k1 k2

function g(k::AbstractVector{T}, times) where T
    u0 = T[ 1.0 ] # NB convert initial values to eltype(k)
    prob = ODEProblem(ode, u0, (0.0, 0.0+maximum(times)), k)
    Array(solve(prob, saveat=times))::Matrix{T}
end

kay = rand(2,50);
MapCols{2}(g, kay, 1:5) # 5 time steps, for each col of parameters

Tracker.gradient(k -> sum(sin, MapCols{2}(g, k, 1:5)), kay)[1]
```

This is quite efficient, and seems to go well with multi-threading:

```julia
@btime MapCols{2}(g, $kay, 1:5)        # 1.394 ms
@btime ThreadMapCols{2}(g, $kay, 1:5)  #   697.333 μs

@btime Tracker.gradient(k -> sum(MapCols{2}(g, k, 1:5)), $kay)[1]       # 2.561 ms
@btime Tracker.gradient(k -> sum(ThreadMapCols{2}(g, k, 1:5)), $kay)[1] # 1.344 ms

Threads.nthreads() == 4 # on my 2/4-core laptop
```

## Elsewhere

Issues about mapslices:
* https://github.com/FluxML/Zygote.jl/issues/92
* https://github.com/FluxML/Flux.jl/issues/741
* https://github.com/JuliaLang/julia/issues/29146

Differential equations:
* https://arxiv.org/abs/1812.01892 "DSAAD"
* http://docs.juliadiffeq.org/latest/analysis/sensitivity.html

Other packages which define gradients of possible interest:
* https://github.com/GiggleLiu/LinalgBackwards.jl
* https://github.com/mcabbott/ArrayAllez.jl

