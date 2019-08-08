
using SliceMap
using Test
using ForwardDiff, Tracker, Zygote, TensorCast, JuliennedArrays

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

    tcm(mat) = @cast out[i,j] := fun(mat[:,j])[i]
    @test res ≈ tcm(mat)
    @test grad ≈ Zygote.gradient(m -> sum(sin, tcm(m)), mat)[1]

    jcols(f,m) = Align(map(f, Slices(m, True(), False())), True(), False())
    @test res ≈ jcols(fun, mat)
    @test grad ≈ Zygote.gradient(m -> sum(sin, jcols(fun, m)), mat)[1]

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

    # tcm3(mat) = @cast out[_,j] := fun(mat[:,j]) # changed here too
    # @test res ≈ tcm3(mat)
    # @test grad ≈ Zygote.gradient(m -> sum(sin, tcm3(m)), mat)[1]

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

    # tcm4(mat) = @cast out[i⊗i′,j] := fun(mat[:,j])[i,i′]  i:3
    # @test res ≈ tcm4(mat)
    # @test grad ≈ Zygote.gradient(m -> sum(sin, tcm4(m)), mat)[1]

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

    # tcm5(mat) = @cast out[i,j] := fun(mat[:,j], 5)[i]
    # @test res ≈ tcm5(mat)
    # @test grad ≈ Zygote.gradient(m -> sum(sin, tcm5(m)), mat)[1]

end
@testset "rows" begin

    mat = randn(4,5)
    fun(x) = 2 .+ x.^2 ./ sum(x)

    res = mapslices(fun, mat, dims=2)
    @test res ≈ maprows(fun, mat)

    grad = ForwardDiff.gradient(m -> sum(sin, mapslices(fun, m, dims=2)), mat)
    @test grad ≈ Tracker.gradient(m -> sum(sin, maprows(fun, m)), mat)[1]
    @test grad ≈ Zygote.gradient(m -> sum(sin, maprows(fun, m)), mat)[1]

    # tcm2(mat) = @cast out[i,j] := fun(mat[i,:])[j]
    # @test res ≈ tcm2(mat)
    # @test grad ≈ Zygote.gradient(m -> sum(sin, tcm2(m)), mat)[1]

    jrows(f,m) = Align(map(f, Slices(m, False(), True())), False(), True())
    @test res ≈ jrows(fun, mat)
    @test grad ≈ Zygote.gradient(m -> sum(sin, jrows(fun, m)), mat)[1]

end
@testset "slices of a 4-tensor" begin

    ten = randn(3,4,5,2)
    fun(x::AbstractVector) = sqrt(3) .+ x.^3 ./ (sum(x)^2)
    res = mapslices(fun, ten, dims=3)

    @test res ≈ slicemap(fun, ten, dims=3)

    grad = ForwardDiff.gradient(x -> sum(sin, slicemap(fun, x, dims=3)), ten)
    @test_broken grad ≈ Zygote.gradient(x -> sum(sin, slicemap(fun, x, dims=3)), ten)[1]

    jthree(f,m) = Align(map(f,
        Slices(m, False(), False(), True(), False())
        ), False(), False(), True(), False())
    @test res ≈ jthree(fun, ten)
    @test grad ≈ Zygote.gradient(m -> sum(sin, jthree(fun, m)), ten)[1]

    j3(f,m) = Align(map(f, Slices(m, 3)), 3)
    @test res ≈ j3(fun, ten)
    @test grad ≈ Zygote.gradient(m -> sum(sin, j3(fun, m)), ten)[1]

end
