# Parameters x, variables y, residuals f
struct ImplicitFunction{matrixfree,F,C,L,T} <: Function
    # A function which takes x as input and returns a tuple (ystar, df/dy) such that f(x, ystar) = 0. df/dy is optional and can be replaced by nothing to compute it via automatic differentiation. Jacobian should only be returned if it's more cheaply available than using AD, e.g. when using BFGS approximation of the Hessian in IPOPT.
    forward::F
    # The conditions function f(x, y) which must be 0 at ystar. Note that variables which don't show up in x and are closed over instead will be assumed to have no effect on the optimal solution. So it's the user's responsibility to ensure x includes all the interesting variables to be differentiated with respect to.
    conditions::C
    # A linear system solver to solve df/dy' \ v
    linear_solver::L
    # The acceptable tolerance for f(x, ystar) to use the implicit function theorem at x
    tol::T
    # A booolean to decide whether or not to error if the tolerance is violated, i.e. norm(f(x, ystar)) > tol. If false, we return a gradient of NaNs.
    error_on_tol_violation::Bool
end
function ImplicitFunction(
    forward::F,
    conditions::C;
    tol::T = 1e-5,
    error_on_tol_violation = false,
    matrixfree = false,
    linear_solver::L = _default_solver(matrixfree),
) where {F,C,L,T}
    return ImplicitFunction{matrixfree,F,C,L,T}(
        forward,
        conditions,
        linear_solver,
        tol,
        error_on_tol_violation,
    )
end

function _default_solver(matrixfree)
    if matrixfree
        return (A, b) -> begin
            L = LinearMap(A, length(b))
            return gmres(L, b)
        end
    else
        return (A, b) -> A \ b
    end
end

(f::ImplicitFunction)(x) = f.forward(x)[1]
(f::ImplicitFunction)() = f.forward()[1]

function ChainRulesCore.rrule(
    rc::RuleConfig,
    f::ImplicitFunction{matrixfree},
    x,
) where {matrixfree}
    ystar, _dfdy = f.forward(x)
    flat_ystar, unflatten_y = flatten(ystar)
    forward_returns_jacobian = _dfdy !== nothing
    if forward_returns_jacobian
        dfdy = _dfdy
        if matrixfree
            # y assumed flat if dfdy is passed in
            pby = v -> dfdy' * v
        else
            pby = nothing
        end
    else
        _conditions_y =
            flat_y -> begin
                return flatten(f.conditions(x, unflatten_y(flat_y)))[1]
            end
        if matrixfree
            dfdy = nothing
            _, _pby = rrule_via_ad(rc, _conditions_y, flat_ystar)
            pby = v -> _pby(v)[2]
        else
            # Change this to AbstractDifferentiation
            dfdy = Zygote.jacobian(_conditions_y, flat_ystar)[1]
            pby = nothing
        end
    end
    _conditions_x = (conditions, x) -> begin
        return flatten(conditions(x, ystar))[1]
    end
    residual, pbx = rrule_via_ad(rc, _conditions_x, f.conditions, x)
    return ystar,
    ∇ -> begin
        if norm(residual) > f.tol && f.error_on_tol_violation
            throw(
                ArgumentError(
                    "The acceptable tolerance for the implicit function theorem is not satisfied for the current problem. Please double check your function definition, increase the tolerance, or set `error_on_tol_violation` to false to ignore the violation and return `NaN`s for the gradient.",
                ),
            )
        end
        if matrixfree
            ∇f, ∇x = Base.tail(pbx(f.linear_solver(pby, -flatten(∇)[1])))
        else
            ∇f, ∇x = Base.tail(pbx(f.linear_solver(dfdy', -flatten(∇)[1])))
        end
        ∇imf = Tangent{typeof(f)}(
            conditions = Tangent{typeof(f.conditions)}(; ChainRulesCore.backing(∇f)...),
        )
        if norm(residual) <= f.tol
            return (∇imf, ∇x)
        else
            return (nanlike(∇imf), nanlike(∇x))
        end
    end
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    f::ImplicitFunction{matrixfree},
) where {matrixfree}
    ystar, _dfdy = f.forward()
    flat_ystar, unflatten_y = flatten(ystar)
    forward_returns_jacobian = _dfdy !== nothing
    if forward_returns_jacobian
        dfdy = _dfdy
        if matrixfree
            # y assumed flat if dfdy is passed in
            pby = v -> dfdy' * v
        else
            pby = nothing
        end
    else
        _conditions_y = flat_y -> begin
            return flatten(f.conditions(unflatten_y(flat_y)))[1]
        end
        if matrixfree
            dfdy = nothing
            _, _pby = rrule_via_ad(rc, _conditions_y, flat_ystar)
            pby = v -> _pby(v)[2]
        else
            # Change this to AbstractDifferentiation
            dfdy = Zygote.jacobian(_conditions_y, flat_ystar)[1]
            pby = nothing
        end
    end
    _conditions = (conditions) -> begin
        return flatten(conditions(ystar))[1]
    end
    residual, pbf = rrule_via_ad(rc, _conditions, f.conditions)
    return ystar,
    ∇ -> begin
        if norm(residual) > f.tol && f.error_on_tol_violation
            throw(
                ArgumentError(
                    "The acceptable tolerance for the implicit function theorem is not satisfied for the current problem. Please double check your function definition, increase the tolerance, or set `error_on_tol_violation` to false to ignore the violation and return `NaN`s for the gradient.",
                ),
            )
        end
        if matrixfree
            ∇f = pbf(f.linear_solver(pby, -flatten(∇)[1]))[2]
        else
            ∇f = pbf(f.linear_solver(dfdy', -flatten(∇)[1]))[2]
        end
        ∇imf = Tangent{typeof(f)}(
            conditions = Tangent{typeof(f.conditions)}(; ChainRulesCore.backing(∇f)...),
        )
        if norm(residual) <= f.tol
            return (∇imf,)
        else
            return (nanlike(∇imf),)
        end
    end
end

function nanlike(x)
    flat, un = flatten(x)
    return un(similar(flat) .= NaN)
end
