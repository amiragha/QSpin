module QSpin

using SparseArrays

import Base: kron, +, -, *

include("spinstate.jl")

export SpinState
export tensorprod

end
