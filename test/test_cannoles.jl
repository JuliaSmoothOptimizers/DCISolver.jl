using CaNNOLeS, NLPModels

# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
stats = cannoles(nls)

# Constrained
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2
                 c=x->[x[1] * x[2] - 1], lcon=[0.0], ucon=[0.0])
stats = cannoles(nls)
