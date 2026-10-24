# Verifies that the R library matches the pinned renv.lock: R version and every package
# version. Prints one JSON object per line and exits non-zero on any mismatch.
#
# Usage: Rscript --vanilla r_env.R path/to/renv.lock

args <- commandArgs(trailingOnly = TRUE)
lock_path <- args[[1]]
failed <- FALSE

emit <- function(check, status, expected = "", generated = "", note = "") {
  esc <- function(x) gsub('"', '\\\\"', gsub("\\\\", "\\\\\\\\", as.character(x)))
  cat(sprintf('{"check": "%s", "status": "%s", "expected": "%s", "generated": "%s", "note": "%s"}\n',
              esc(check), status, esc(expected), esc(generated), esc(note)))
  if (status == "FAIL") failed <<- TRUE
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  emit("R packages vs renv.lock", "FAIL", note = "jsonlite is not installed")
  quit(status = 1)
}
lock <- jsonlite::fromJSON(lock_path, simplifyVector = FALSE)

r_now <- paste(R.version$major, R.version$minor, sep = ".")
emit("R version vs renv.lock", if (identical(r_now, lock$R$Version)) "PASS" else "FAIL",
     expected = lock$R$Version, generated = r_now)

# The DESCRIPTION Version field as written (e.g. "2.1-1"), which is what renv.lock records;
# packageVersion() would print it as "2.1.1".
installed_version <- function(name) {
  v <- suppressWarnings(utils::packageDescription(name, fields = "Version"))
  if (is.null(v) || is.na(v)) "not installed" else v
}

bad <- character(0)
for (pkg in lock$Packages) {
  have <- installed_version(pkg$Package)
  if (!identical(have, pkg$Version)) bad <- c(bad, sprintf("%s %s (locked %s)", pkg$Package, have, pkg$Version))
}
n <- length(lock$Packages)
emit("R packages vs renv.lock", if (length(bad) == 0) "PASS" else "FAIL",
     expected = sprintf("%d pinned packages", n), generated = sprintf("%d match", n - length(bad)),
     note = paste(bad, collapse = "; "))

fc <- tryCatch(as.character(utils::packageVersion("forecast")), error = function(e) NA_character_)
emit("forecast package", if (is.na(fc)) "FAIL" else "PASS", generated = if (is.na(fc)) "not installed" else fc)
quit(status = if (failed) 1 else 0)
