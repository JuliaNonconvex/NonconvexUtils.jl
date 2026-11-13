using Documenter
using NonconvexUtils

makedocs(;
    sitename="NonconvexUtils.jl",
    modules=[NonconvexUtils],
    source="src",
    build="build",
    doctest=false,
    warnonly=true,
    remotes=nothing,
    format=Documenter.HTML(;
        prettyurls=true, edit_link=nothing, repolink=nothing, inventory_version=""
    ),
    pages=["Home" => "index.md"],
)
