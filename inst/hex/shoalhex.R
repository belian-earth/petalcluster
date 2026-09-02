# The shoal hex sticker, in the house style shared with a5R: black hexagon,
# sage border, a light thin wordmark. The motif is a shoal of fish, and the
# colours are earned: a point cloud is clustered with shoal_hdbscan() and
# each fish takes its cluster's colour from shoal_palette(), with the noise
# points in grey.
#
# Run from the package root, with the package installed:
#   Rscript inst/hex/shoalhex.R
# Draws the sticker and hands it to usethis::use_logo(), which places the
# resized copy at man/figures/logo.png. pkgdown builds the favicons from it
# on the next local site build; after changing the sticker, refresh them
# with pkgdown::build_favicons(overwrite = TRUE). Needs rsvg and usethis.
# Not shipped (see .Rbuildignore).

library(shoal)
# use_logo() asks before replacing an existing logo, and under Rscript the
# answer is silently no; this is the regenerate script, so always replace.
options(usethis.overwrite = TRUE)

sage <- "#74ac90"
w <- 240
h <- 277
cx <- w / 2
cy <- h / 2

# Pointy-top hexagon filling the 240 x 277 canvas, as a5R's does.
ang <- seq(90, 390, by = 60) * pi / 180
hx <- cx + (w / 2 - 3) * cos(ang) / cos(30 * pi / 180)
hy <- cy - (h / 2 - 3) * sin(ang)
hex <- paste(sprintf("%.1f,%.1f", hx, hy), collapse = " ")

# -- The point cloud: four shoals, each elongated along its own heading,
#    plus scattered strays. ---------------------------------------------------
set.seed(7)
shoals <- data.frame(
  x = c(64, 176, 74, 182),
  y = c(72, 88, 212, 204),
  heading = c(-20, 160, 30, 200),
  n = c(34, 30, 32, 30)
)
pts <- do.call(
  rbind,
  lapply(seq_len(nrow(shoals)), function(j) {
    s <- shoals[j, ]
    along <- rnorm(s$n, 0, 22)
    across <- rnorm(s$n, 0, 9)
    th <- s$heading * pi / 180
    data.frame(
      x = s$x + along * cos(th) - across * sin(th),
      y = s$y + along * sin(th) + across * cos(th)
    )
  })
)
strays <- data.frame(x = runif(40, 30, w - 30), y = runif(40, 40, h - 40))
pts <- rbind(pts, strays)

# -- Cluster with the package: the colours come from the result. -------------
fit <- shoal_hdbscan(as.matrix(pts), min_cluster_size = 12L, min_samples = 4L)
stopifnot(fit$n_clusters == 4L)
pal <- shoal_palette(fit$n_clusters)
col <- ifelse(is.na(fit$cluster), "#8f9a95", pal[fit$cluster])

# Each fish swims along its cluster's principal axis, with a little jitter;
# strays face wherever they like.
heading <- vapply(
  seq_len(nrow(pts)),
  function(i) {
    cl <- fit$cluster[i]
    if (is.na(cl)) {
      return(runif(1, 0, 360))
    }
    members <- as.matrix(pts[!is.na(fit$cluster) & fit$cluster == cl, ])
    axis <- prcomp(members)$rotation[, 1]
    base <- atan2(axis[2], axis[1]) * 180 / pi
    # prcomp's sign is arbitrary; keep the direction the shoal was laid out in.
    if (abs(((base - shoals$heading[cl] + 180) %% 360) - 180) > 90) {
      base <- base + 180
    }
    base + rnorm(1, 0, 9)
  },
  numeric(1)
)
size <- ifelse(
  is.na(fit$cluster),
  runif(nrow(pts), 0.55, 0.75),
  runif(nrow(pts), 0.7, 1.05)
)
opacity <- ifelse(is.na(fit$cluster), 0.55, 0.92)

fish <- sprintf(
  '<use href="#fish" transform="translate(%.1f %.1f) rotate(%.1f) scale(%.2f)" fill="%s" fill-opacity="%.2f"/>',
  pts$x,
  pts$y,
  heading,
  size,
  col,
  opacity
)

svg <- sprintf(
  '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">
<defs>
  <!-- A small fish pointing along +x: body, tail, and a dark eye. -->
  <symbol id="fish" overflow="visible">
    <path d="M -3.6 0 C -2.4 -2.4 2.2 -2.6 4.6 0 C 2.2 2.6 -2.4 2.4 -3.6 0 Z"/>
    <path d="M -3.2 0 L -6.2 -2.3 L -5.4 0 L -6.2 2.3 Z"/>
    <circle cx="2.9" cy="-0.5" r="0.55" fill="#000000" fill-opacity="0.7"/>
  </symbol>
  <clipPath id="hex"><polygon points="%s"/></clipPath>
</defs>
<polygon points="%s" fill="#000000" stroke="%s" stroke-width="2.5" stroke-linejoin="round"/>
<g clip-path="url(#hex)">
%s
</g>
<text x="%.1f" y="%.1f" text-anchor="middle" font-family="Inter, Lato, Helvetica, Arial, sans-serif" font-weight="200" font-size="54" fill="#dfe6e2" letter-spacing="1.5">shoal</text>
</svg>',
  w,
  h,
  w,
  h,
  hex,
  hex,
  sage,
  paste(fish, collapse = "\n"),
  cx,
  cy + 19
)

# Render at five times the sticker size; use_logo() scales it back down.
svg_path <- tempfile(fileext = ".svg")
png_path <- tempfile(fileext = ".png")
writeLines(svg, svg_path)
rsvg::rsvg_png(svg_path, png_path, width = 5L * w, height = 5L * h)

usethis::use_logo(png_path, geometry = sprintf("%dx%d", w, h))
