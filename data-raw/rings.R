## code to prepare `rings` dataset goes here
# -- Generate concentric rings with noise --
set.seed(5446)
ring <- function(n, radius, sd = 0.08) {
  theta <- runif(n, 0, 2 * pi)
  r <- rnorm(n, radius, sd)
  cbind(x = r * cos(theta), y = r * sin(theta))
}
rings <- rbind(
  ring(300, radius = 0.5),
  ring(400, radius = 1.2),
  ring(500, radius = 2.0),
  cbind(x = runif(60, -2.5, 2.5), y = runif(60, -2.5, 2.5))
)
usethis::use_data(rings, overwrite = TRUE)
