using Pkg: Pkg
using NormalizingFlowFilters
using Test
using TestReports
using TestReports.EzXML
using Aqua
using Documenter

report_testsets = @testset ReportingTestSet "" begin
    @info "Testing code quality with Aqua.jl."
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NormalizingFlowFilters; ambiguities=false)
        Aqua.test_ambiguities(NormalizingFlowFilters)
    end

    @info "Running package tests."
    include("test_assimilate_data.jl")
    include("test_conditional_linear.jl")

    # Set metadata for doctests.
    DocMeta.setdocmeta!(
        NormalizingFlowFilters,
        :DocTestSetup,
        :(using NormalizingFlowFilters, Test);
        recursive=true,
    )

    # Run doctests.
    @info "Running doctests."
    doctest(NormalizingFlowFilters; manual=true)
end

xml_all = report(report_testsets)
outputfilename = joinpath(@__DIR__, "..", "report.xml")
open(outputfilename, "w") do fh
    print(fh, xml_all)
end

@test !any_problems(report_testsets)
