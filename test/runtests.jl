
using SliceMap
using Test
using ForwardDiff, Tracker, Zygote, JuliennedArrays

Zygote.refresh()

@testset "columns" begin

    mat = rand(1:9, 3,10)
    fun(x) = 2 .+ x.^2
    res = mapslices(fun, mat, dims=1)

    @test res ≈ mapcols(fun, mat)
    @test res ≈ MapCols{3}(fun, mat)
    @test res ≈ MapCols(fun, mat)

    @test res ≈ tmapcols(fun, mat)
    @test res ≈ ThreadMapCols{3}(fun, mat)
    @test res ≈ ThreadMapCols(fun, mat)

    grad = ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat)

    @test grad ≈ Tracker.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, MapCols(fun, m)), mat)[1]

    @test grad ≈ Tracker.gradient(m -> sum(sin, tmapcols(fun, m)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, ThreadMapCols{3}(fun, m)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, ThreadMapCols(fun, m)), mat)[1]

    @test grad ≈ Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]

    @test grad ≈ Zygote.gradient(m -> sum(sin, tmapcols(fun, m)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, ThreadMapCols{3}(fun, m)), mat)[1]

    jcols(f,m) = Align(map(f, JuliennedArrays.Slices(m, True(), False())), True(), False())
    @test res ≈ jcols(fun, mat)
    @test grad ≈ Zygote.gradient(m -> sum(sin, jcols(fun, m)), mat)[1]

    # Case of length 1, was an error for _threadmap
    @test res[:,1:1] ≈ mapcols(fun, mat[:, 1:1])
    @test res[:,1:1] ≈ tmapcols(fun, mat[:, 1:1])
    @test res[:,1:1] ≈ ThreadMapCols(fun, mat[:, 1:1])

end
@testset "columns -> scalar" begin

    mat = rand(1:9, 3,10)
    fun(x) = sum(x) # different function!
    res = mapslices(fun, mat, dims=1)

    @test res ≈ mapcols(fun, mat)
    @test res ≈ MapCols{3}(fun, mat)

    grad = ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=1)), mat)

    @test grad ≈ Tracker.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]

    @test grad ≈ Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]

end
@testset "columns -> matrix" begin

    mat = rand(1:9, 3,10)
    fun(x) = x .* x' # different function! vector -> matrix
    res = mapslices(vec∘fun, mat, dims=1)

    @test res ≈ mapcols(fun, mat)
    @test res ≈ MapCols{3}(fun, mat)

    grad = ForwardDiff.gradient(m -> sum(sin, mapslices(vec∘fun, m, dims=1)), mat)

    @test grad ≈ Tracker.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]

    @test grad ≈ Zygote.gradient(m -> sum(sin, mapcols(fun, m)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m)), mat)[1]

end
@testset "columns w args" begin

    mat = randn(Float32, 3,10)
    fun(x, s) = 1 .+ x .* s
    res = mapslices(x -> vec(fun(x,5)), mat, dims=1)

    @test res ≈ mapcols(fun, mat, 5)
    @test res ≈ MapCols{3}(fun, mat, 5)

    grad = ForwardDiff.gradient(m -> sum(sin, mapslices(x -> vec(fun(x,5)), m, dims=1)), mat)

    @test grad ≈ Tracker.gradient(m -> sum(sin, mapcols(fun, m, 5)), mat)[1]
    @test grad ≈ Tracker.gradient(m -> sum(sin, MapCols{3}(fun, m, 5)), mat)[1]

    @test grad ≈ Zygote.gradient(m -> sum(sin, mapcols(fun, m, 5)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, MapCols{3}(fun, m, 5)), mat)[1]

end
@testset "rows" begin

    mat = randn(4,5)
    fun(x) = 2 .+ x.^2 ./ sum(x)

    res = mapslices(fun, mat, dims=2)
    @test res ≈ maprows(fun, mat)

    grad = ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=2)), mat)
    @test grad ≈ Tracker.gradient(m -> sum(sin, maprows(fun, m)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, maprows(fun, m)), mat)[1]

    jrows(f,m) = Align(map(f, JuliennedArrays.Slices(m, False(), True())), False(), True())
    @test res ≈ jrows(fun, mat)
    @test grad ≈ Zygote.gradient(m -> sum(sin, jrows(fun, m)), mat)[1]

end
@testset "slices of a 4-tensor" begin

    ten = randn(3,4,5,2)
    fun(x::AbstractVector) = sqrt(3) .+ x.^3 ./ (sum(x)^2)
    res = mapslices(fun, ten, dims=3)

    @test res ≈ slicemap(fun, ten, dims=3)

    grad = ForwardDiff.gradient(x -> sum(sin, slicemap(fun, x, dims=3)), ten)
    @test grad ≈ Zygote.gradient(x -> sum(sin, slicemap(fun, x, dims=3)), ten)[1]

    jthree(f,m) = Align(map(f,
        JuliennedArrays.Slices(m, False(), False(), True(), False())
        ), False(), False(), True(), False())
    @test res ≈ jthree(fun, ten)
    @test grad ≈ Zygote.gradient(m -> sum(sin, jthree(fun, m)), ten)[1]

    j3(f,m) = Align(map(f, JuliennedArrays.Slices(m, 3)), 3)
    @test res ≈ j3(fun, ten)
    @test grad ≈ Zygote.gradient(m -> sum(sin, j3(fun, m)), ten)[1]

end
@testset "gradient of the function" begin

    struct F W end
    (f::F)(x) = f.W * x # toy version of e.g. Flux.Dense
    w = rand(3,2)
    x = rand(2,5)
    gradx = ForwardDiff.gradient(x -> sum(mapslices(F(w), x, dims=1)), x)
    gradw = ForwardDiff.gradient(w -> sum(mapslices(F(w), x, dims=1)), w)

    wp = Tracker.param(w)
    xp = Tracker.param(x)
    Tracker.back!(sum(mapcols(F(wp), xp)))
    @test Tracker.grad(xp) ≈ gradx
    @test Tracker.grad(wp) == 0 .* gradw # bug or a feature?

    # fp = F(wp)
    # wp.grad .= 0; xp.grad .= 0;
    # Tracker.back!(sum(mapcols(fp, xp)))
    # @test Tracker.grad(xp) ≈ gradx
    # @test_broken Tracker.grad(wp) ≈ gradw # zero

    f = F(w)
    grad_mapcols = Zygote.gradient(() -> sum(mapcols(f, x)), Zygote.Params([w,x]))
    @test grad_mapcols[x] ≈ gradx
    @test grad_mapcols[w] == nothing # bug or a feature?

    grad_slicemap = Zygote.gradient(() -> sum(slicemap(f, x, dims=1)), Zygote.Params([w,x]))
    @test grad_slicemap[x] ≈ gradx
    @test grad_slicemap[w] ≈ gradw
    @test gradw ≈ Zygote.gradient(w -> sum(slicemap(F(w), x, dims=1)), w)[1]
    # Using F(w) with Params() gives wrong answers:
    # https://github.com/FluxML/Zygote.jl/issues/522#issuecomment-605935652

end
@testset "slicemap rev=true" begin

    rec(store) = x -> (push!(store, first(x)); x)
    A = [1,1,1] .* (1:5)'

    store = []
    slicemap(rec(store), A; dims=1)
    @test store == 1:5

    store = []
    slicemap(rec(store), A; dims=1, rev=true)
    @test store == 5:-1:1

    # gradient check as above
    ten = randn(3,4,5,2)
    fun(x::AbstractVector) = sqrt(3) .+ x.^3 ./ (sum(x)^2)
    res = mapslices(fun, ten, dims=3)
    @test res ≈ slicemap(fun, ten; dims=3, rev=true)
    @test res ≈ slicemap(fun, ten; dims=3, rev=false)

    grad = ForwardDiff.gradient(x -> sum(sin, mapslices(fun, x, dims=3)), ten)
    @test grad ≈ Zygote.gradient(x -> sum(sin, slicemap(fun, x, dims=3, rev=false)), ten)[1]
    @test grad ≈ Zygote.gradient(x -> sum(sin, slicemap(fun, x, dims=3, rev=true)), ten)[1]

end
