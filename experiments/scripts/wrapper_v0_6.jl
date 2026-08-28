# Frozen copy of the production forecasting wrapper, as it ACTUALLY RAN.
#
# ####################################################################
# # THIS FILE IS A RECONSTRUCTION FROM AN UNCOMMITTED WORKING TREE.  #
# ####################################################################
#
# "v0_6" is a CAMPAIGN LABEL, not a configuration. In the harness repository it is a
# function that calls whatever package is installed; the configuration below is the only
# faithful definition of what that arm meant. Rewriting its keywords by hand elsewhere is
# exactly how fidelity is lost without anyone noticing, so campaign D calls this wrapper
# rather than reimplementing it.
#
# The problem this banner exists for: in the harness repository, this wrapper carried
# UNCOMMITTED modifications from shortly after the last harness commit until consolidation.
# Every campaign that invoked it in that window — including the published isolation
# campaign and the v0_6 production run — used the MODIFIED version, while the committed
# version still said something else. The difference is the short-series trigger:
#
#     committed in the harness repo   curta = length(y) <= 150          (fixed threshold)
#     what actually ran               curta = length(y) <= 5 * (5 + 2s) (relative)
#
# The relative form is reproduced below because it is what produced the numbers. For s = 1
# the two forms differ sharply (35 against 150), so on yearly and weekly the choice is not
# cosmetic.
#
# The provenance stamp did not catch this: it checked only the package tree, not the
# harness tree, so every affected output reads "clean". `provenance.jl` in this directory
# checks both.
module WrapperV06

using DataFrames, TimeSeries
import Sarimax

export forecastV06

"""
    v06Config(y, s) -> Dict{Symbol,Any}

The v0_6 production configuration, argument by argument, with the short-series branch
resolved for the given series.
"""
function v06Config(y::Vector{Float64}, s::Int)
    # The trigger is RELATIVE to the common conditioning, not a fixed threshold. `auto`
    # imposes searchLb = conditioningLags(maxp, maxq, maxP, maxQ, s) = 5 + 2s under the
    # default order bounds, i.e. 7 at s=1, 13 at s=4, 29 at s=12, 53 at s=24. It is that
    # quantity which consumes the sample, so the point at which :free stops paying tracks
    # searchLb rather than any absolute number.
    #
    # A fixed threshold of 150 would mark essentially every yearly series as short (median
    # T = 29), applying :free where the conditioning consumes 24% of the sample rather than
    # the 43% that motivated the change on monthly — paying 1.34x in cost without the
    # problem it solves. The factor 5 reproduces the 145-150 threshold used on monthly.
    searchLb = 5 + 2 * s
    short = length(y) <= 5 * searchLb
    Dict{Symbol,Any}(
        :seasonality                     => s,
        :objectiveFunction               => "mse",
        # the short-series branch: the two keywords below are the whole of it
        :initialization                  => short ? :free : :zeroed,
        :warmStartFromBox                => short,
        :maxTimeSeconds                  => short ? 120.0 : nothing,
        :seasonalForm                    => :multiplicative,
        :stationary                      => true,
        :stationarityMargin              => 1e-6,
        :invertible                      => false,
        :invertibilityMargin             => 1e-6,
        :assertStationarity              => true,
        :assertInvertibility             => true,
        :rootMargin                      => 1e-2,
        :constrainedRefit                => false,
        :searchMethod                    => "stepwise",
        :informationCriteria             => "aicc",
        :integrationTest                 => "kpssShort",
        :seasonalIntegrationTest         => "seas",
        :d                               => -1,
        :D                               => -1,
        :maxd                            => 2,
        :maxD                            => 1,
        :maxp                            => 5,
        :maxq                            => 5,
        :maxP                            => 2,
        :maxQ                            => 2,
        :maxOrder                        => 5,
        :multistart                      => false,
        :parallel                        => false,
        :cvarLevel                       => 0.9,
        :outlierDetection                => false,
        # Turned on UNCONDITIONALLY across all five frequencies by this wrapper. Measured
        # to cost +0.043 OWA on weekly, where no series with d+D>=2 ever selects the
        # bare (0,d,0)(0,D,0) order the guard exists to block — so on that frequency it
        # prevents nothing and acts only by removing one of the stepwise seed models,
        # which redirects the search. Kept here because it is what ran.
        :requireTermsWhenOverDifferenced => true,
        :requireMAWhenDoublyDifferenced  => false,
    )
end

"""
    forecastV06(y, s, H, S) -> (prediction, scenarios)

Point forecast plus `S` simulated scenarios, matching the production signature.
"""
function forecastV06(y::Vector{Float64}, s::Int, H::Int, S::Int)
    dataset = Sarimax.loadDataset(DataFrame(y = y))
    model = Sarimax.auto(dataset; v06Config(y, s)...)
    Sarimax.predict!(model; stepsAhead = H)
    scenarios = Sarimax.simulate(model, H, S)
    prediction = Vector{Float64}(TimeSeries.values(model.forecast))
    simulated = permutedims(hcat(values(scenarios)...))'
    (prediction, simulated)
end

end # module
