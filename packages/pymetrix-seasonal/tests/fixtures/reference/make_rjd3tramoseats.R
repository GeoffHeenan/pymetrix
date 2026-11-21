#!/usr/bin/env Rscript
# TRAMO-SEATS as Eurostat's own R interface to JDemetra+ (rjd3tramoseats) computes it, for tests/test_tramo_seats.py:
# five of R's datasets, each with JDemetra+'s seven predefined specifications (RSA0 to RSAfull). It records the series,
# the adjusted series, the model TRAMO chose, logs or levels, and the regressors (outliers among them), with the
# versions of R, Java and the rjd3 packages.
#
#   Rscript packages/pymetrix-seasonal/tests/fixtures/reference/make_rjd3tramoseats.R [out.json]
#
# Needs Java 21, rJava and rjd3tramoseats (install.packages("rjd3tramoseats")) and jsonlite. The CI workflow
# .github/workflows/jdemetra-reference.yml runs it on every push, then pymetrix on the same cases; saved as
# packages/pymetrix-seasonal/tests/fixtures/reference/rjd3tramoseats.json (the default), the tests use it wherever they run.

suppressPackageStartupMessages({
  library(rjd3toolkit)
  library(rjd3tramoseats)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[[1]] else "packages/pymetrix-seasonal/tests/fixtures/reference/rjd3tramoseats.json"

series <- list(
  AirPassengers = datasets::AirPassengers,    # monthly, 1949-1960: the airline model's series
  UKgas = datasets::UKgas,                    # quarterly, 1960-1986
  USAccDeaths = datasets::USAccDeaths,        # monthly, 1973-1978: short
  nottem = datasets::nottem,                  # monthly temperatures, 1920-1939: additive
  UKDriverDeaths = datasets::UKDriverDeaths   # monthly, 1969-1984: the 1983 seat-belt law, a level shift
)
specs <- c("rsa0", "rsa1", "rsa2", "rsa3", "rsa4", "rsa5", "rsafull")
orders <- c("arima.p", "arima.d", "arima.q", "arima.bp", "arima.bd", "arima.bq")
keys <- c("sa", "log", orders, "regression.details.description")

cases <- list()
for (name in names(series)) {
  for (spec in specs) {
    m <- rjd3tramoseats::tramoseats(series[[name]], spec, userdefined = keys)
    u <- m$user_defined
    if (!length(cases)) str(u)  # what the first case returned, for the log
    cases[[length(cases) + 1]] <- list(
      series = name,
      spec = spec,
      sa = signif(as.numeric(u[["sa"]]), 15),
      log = isTRUE(as.logical(u[["log"]])),
      model = vapply(orders, function(k) as.integer(u[[k]]), integer(1), USE.NAMES = FALSE),
      regressors = as.character(unlist(u[["regression.details.description"]]))
    )
  }
}

java <- rJava::.jcall("java/lang/System", "S", "getProperty", "java.version")
packages <- vapply(c("rjd3toolkit", "rjd3tramoseats", "rJava"), function(p) as.character(packageVersion(p)), "")
data <- lapply(series, function(y) list(period = frequency(y), start = as.integer(start(y)), values = as.numeric(y)))
writeLines(toJSON(list(
  source = "rjd3tramoseats::tramoseats(y, spec), Eurostat's R interface to JDemetra+; made by make_rjd3tramoseats.R",
  made = format(Sys.Date()), R = R.version.string, java = java, packages = as.list(packages),
  series = data, cases = cases
), auto_unbox = TRUE, digits = NA), out)
cat("wrote", length(cases), "cases to", out, "\n")
