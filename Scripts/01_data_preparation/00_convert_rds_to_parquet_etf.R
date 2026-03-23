if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow", repos = "https://cloud.r-project.org")
}

library(arrow)

input_dir <- "data_raw/intraday_etfs"
output_dir <- "data_intermediate/converted_parquet_etf"

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