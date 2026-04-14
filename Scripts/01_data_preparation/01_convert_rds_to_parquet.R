"
Convert Stock RDS Files To Parquet

Purpose:
  Convert raw daily stock RDS files to Parquet while keeping a one-file-per-day structure.

Inputs:
  - Daily stock files in data_raw/intraday_stocks.

Outputs:
  - Converted files in data_intermediate/converted_parquet.

Main Steps:
  - Loop over each RDS file.
  - Normalize the date column.
  - Write one Parquet file per input file.
"
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow", repos = "https://cloud.r-project.org")
}

library(arrow)

input_dir <- "data_raw/intraday_stocks"
output_dir <- "data_intermediate/converted_parquet"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(input_dir, pattern = "\\.rds$", full.names = TRUE)

for (i in seq_along(files)) {
  file_path <- files[i]
  cat(sprintf("[%d/%d] Converting %s\n", i, length(files), basename(file_path)))

  x <- readRDS(file_path)

  x$date <- as.Date(x$date, format = "%Y%m%d")

  out_path <- file.path(
    output_dir,
    sub("\\.rds$", ".parquet", basename(file_path))
  )

  write_parquet(x, out_path)
}
