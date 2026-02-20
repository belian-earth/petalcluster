check_numeric_matrix <- function(x, call = rlang::caller_env()) {
  if (is.data.frame(x)) {
    numeric_cols <- vapply(x, is.numeric, logical(1L))
    if (sum(numeric_cols) < 2L) {
      cli::cli_abort(
        "{.arg x} data frame must have at least 2 numeric columns.",
        call = call
      )
    }
    x <- as.matrix(x[, numeric_cols, drop = FALSE])
  }
  if (!is.matrix(x) || !is.numeric(x)) {
    cli::cli_abort("{.arg x} must be a numeric matrix or data frame.", call = call)
  }
  if (nrow(x) < 1L) {
    cli::cli_abort("{.arg x} must have at least 1 row.", call = call)
  }
  if (ncol(x) < 2L) {
    cli::cli_abort("{.arg x} must have at least 2 columns.", call = call)
  }

  # Drop rows containing NA — NaN poisons distance calculations in Rust
  incomplete <- !stats::complete.cases(x)
  if (any(incomplete)) {
    n_drop <- sum(incomplete)
    cli::cli_warn(
      c("Removed {n_drop} row{?s} containing missing values.",
        "i" = "{nrow(x) - n_drop} complete row{?s} remaining."),
      call = call
    )
    x <- x[!incomplete, , drop = FALSE]
    if (nrow(x) < 1L) {
      cli::cli_abort("No complete rows remaining after removing {.val NA}s.", call = call)
    }
  }
  x
}

check_positive_number <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (!rlang::is_scalar_double(x) && !rlang::is_scalar_integer(x)) {
    cli::cli_abort("{.arg {arg}} must be a single number.", call = call)
  }
  if (is.na(x) || x <= 0) {
    cli::cli_abort("{.arg {arg}} must be positive.", call = call)
  }
  invisible(x)
}

check_positive_integer <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (!rlang::is_scalar_integerish(x)) {
    cli::cli_abort("{.arg {arg}} must be a single integer.", call = call)
  }
  if (is.na(x) || x < 1L) {
    cli::cli_abort("{.arg {arg}} must be a positive integer.", call = call)
  }
  invisible(x)
}
