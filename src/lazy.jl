struct LazyJacobian{symmetric,J1,J2}
    jvp::J1
    jtvp::J2
end
function LazyJacobian(; jvp = nothing, jtvp = nothing, symmetric = false)
    return LazyJacobian{symmetric}(jvp, jtvp)
end
function LazyJacobian{symmetric}(jvp = nothing, jtvp = nothing) where {symmetric}
    if jvp === jtvp === nothing
        throw(ArgumentError("Both the jvp and jtvp operators cannot be nothing."))
    end
    if symmetric
        if jvp !== nothing
            _jtvp = _jvp = jvp
        else
            _jvp = _jtvp = jtvp
        end
    else
        _jvp = jvp
        _jtvp = jtvp
    end
    return LazyJacobian{symmetric,typeof(_jvp),typeof(_jtvp)}(_jvp, _jtvp)
end

struct LazyJacobianTransposed{J}
    j::J
end

LinearAlgebra.adjoint(j::LazyJacobian{false}) = LazyJacobianTransposed(j)
LinearAlgebra.transpose(j::LazyJacobian{false}) = LazyJacobianTransposed(j)
LinearAlgebra.adjoint(j::LazyJacobian{true}) = j
LinearAlgebra.transpose(j::LazyJacobian{true}) = j
LinearAlgebra.adjoint(j::LazyJacobianTransposed) = j.j
LinearAlgebra.transpose(j::LazyJacobianTransposed) = j.j

LinearAlgebra.:*(j::LazyJacobian, v::AbstractVecOrMat) = j.jvp(v)
LinearAlgebra.:*(v::AbstractVecOrMat, j::LazyJacobian) = j.jtvp(v')'
LinearAlgebra.:*(j::LazyJacobianTransposed, v::AbstractVecOrMat) = (v' * j')'
LinearAlgebra.:*(v::AbstractVecOrMat, j::LazyJacobianTransposed) = (j' * v')'
