using Main.DCI, NLPModels, CUTEst

function test_dci(S :: String)
    nlp = ADNLPModel(
        x->0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2,
        [2.0; 2.0; 2.0],
        x->[x[1] + x[3]^2 + 1.0],
        zeros(1),
        zeros(1)
    )
    nlp = CUTEstModel(S)
    try
        output = dci(nlp, nlp.meta.x0, max_time=Inf, max_eval=3000)
        println(output)
    catch ex
        @show ex
    finally
        finalize(nlp)
    end
end

test_dci("MSS1")
