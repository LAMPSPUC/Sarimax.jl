# The 46 instances of the August 2026 certification batch, written as data.
#
# The batch's harness (results/historical/certification-2026-08/received/harness/
# scip_prog_lote2_22-08_1435.jl) defines its instances in casos(): the data-generating
# coefficients come from sorteio() (MersenneTwister seeded by Base.hash of the cell, draw and
# seasonal orders) and the series from gera() (MersenneTwister(1000 + id)). The batch ran on
# Julia 1.12.3. MersenneTwister's seeding changed in Julia 1.11, so under the replication
# environment (Julia 1.10.12) the same code draws other series. This script runs the harness's
# own data code, loaded verbatim, and writes every instance with its coefficients and series, so
# that the runner reads the batch's data under any Julia version.
#
# Run it with Julia 1.11 or later; it needs only the standard library:
#   julia +1.11 certification_2026_08_instances.jl HARNESS.jl OUT.csv
# The committed file (experiments/configs/certification_2026_08_instances.csv) was written by
# Julia 1.11.9. It reproduces the Ipopt objective of every instance whose objective the batch
# recorded (ids 5, 26, 27, 33-38, 44, 45) to the printed precision; see
# reports/certification_batch_2026_08.md. The three real M4 series (ids 28-30) are not
# written: the runner reads them from the M4 training file, as the harness did.

using Random, Statistics

VERSION >= v"1.11" || error("Julia $(VERSION): MersenneTwister draws other numbers before 1.11; run with Julia 1.11 or later")
length(ARGS) == 2 || error("usage: julia certification_2026_08_instances.jl HARNESS.jl OUT.csv")
const HARNESS, OUT = ARGS

src = read(HARNESS, String)
# the data section of the harness: constants, Caso, sorteio, casos, gera (up to the M4 reader)
a = findfirst("const s  = 12", src)
b = findfirst("const TRAIN", src)
(a === nothing || b === nothing) && error("$HARNESS: data section not found")
include_string(Main, src[a.start:b.start-1], basename(HARNESS))

open(OUT, "w") do io
    println(io, "id,cell,label,P,Q,T,draw,mode,sid,phi,theta,Phi,Theta,y")
    for c in casos()
        y = c.sid > 0 ? "" : join(repr.(gera(c)), ";")
        par = c.sid > 0 ? ("", "", "", "") : repr.(c.par)
        println(io, join((c.id, c.cel, c.rotulo, c.P, c.Q, c.sid > 0 ? "" : c.T, c.rep, c.modo, c.sid,
                          par..., y), ","))
    end
end
println("Julia $(VERSION): $(length(casos())) instances written to $OUT")
