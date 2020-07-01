using Base.Test
using NLPModels, CUTEst

include("dci.jl")

function test_dci()
    nlp = ADNLPModel(x->0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2, [2.0; 2.0; 2.0],
                     x->[x[1] + x[3]^2 + 1.0], zeros(1), zeros(1))
    x, fx, dual, primal, eltime, status = dci(nlp, verbose=true)

    println("status = $status")
end

test_dci()
