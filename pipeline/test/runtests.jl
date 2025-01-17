using DrWatson, Test
quickactivate(@__DIR__(), "EpiAwarePipeline")

# Run tests
include("pipeline/test_pipelinetypes.jl");
include("constructors/test_constructors.jl");
include("simulate/test_TruthSimulationConfig.jl");
include("simulate/test_SimulationConfig.jl");
include("infer/test_InferenceConfig.jl");
