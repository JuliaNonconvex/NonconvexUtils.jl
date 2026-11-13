module NonconvexUtils

# NonconvexUtils has been absorbed into NonconvexCore as of v1.2.0. This
# package now only re-exports NonconvexCore so that existing `using
# NonconvexUtils` code keeps working. New code should `using NonconvexCore`
# directly.

using Reexport: @reexport
@reexport using NonconvexCore

end
