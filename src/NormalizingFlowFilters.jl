
"""
This module provides an interface to use conditional normalizing flow for data assimilation.
"""
module NormalizingFlowFilters

include("options.jl")
include("types.jl")
include("assimilate_data.jl")
include("train.jl")
include("layer_conditional_linear.jl")
include("network_linear.jl")

end # module NormalizingFlowFilters
