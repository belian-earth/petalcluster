# The shoal hex sticker, in the house style shared with a5R: black hexagon,
# sage border, a light thin wordmark. The motif is a shoal of fish, and the
# colours are earned: a point cloud is clustered with shoal_hdbscan() and
# each fish takes its cluster's colour from shoal_palette(), with the noise
# points in grey.
#
# Run from the package root: Rscript inst/hex/shoalhex.R
# Writes man/figures/logo.png and the favicon set under pkgdown/favicon/.
# Needs rsvg and magick, and the package itself via load_all().
# Not shipped (see .Rbuildignore).

devtools::load_all(quiet = TRUE)

out <- "pkgdown/favicon"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

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
  x = c(70, 170, 82, 176), y = c(78, 92, 205, 200),
  heading = c(-20, 160, 30, 200), n = c(34, 30, 32, 30)
)
pts <- do.call(rbind, lapply(seq_len(nrow(shoals)), function(j) {
  s <- shoals[j, ]
  along <- rnorm(s$n, 0, 16)
  across <- rnorm(s$n, 0, 6)
  th <- s$heading * pi / 180
  data.frame(
    x = s$x + along * cos(th) - across * sin(th),
    y = s$y + along * sin(th) + across * cos(th)
  )
}))
strays <- data.frame(x = runif(40, 30, w - 30), y = runif(40, 40, h - 40))
pts <- rbind(pts, strays)

# -- Cluster with the package: the colours come from the result. -------------
fit <- shoal_hdbscan(as.matrix(pts), min_cluster_size = 12L, min_samples = 4L)
stopifnot(fit$n_clusters == 4L)
pal <- shoal_palette(fit$n_clusters)
col <- ifelse(is.na(fit$cluster), "#8f9a95", pal[fit$cluster])

# Each fish swims along its cluster's principal axis, with a little jitter;
# strays face wherever they like.
heading <- vapply(seq_len(nrow(pts)), function(i) {
  cl <- fit$cluster[i]
  if (is.na(cl)) return(runif(1, 0, 360))
  members <- as.matrix(pts[!is.na(fit$cluster) & fit$cluster == cl, ])
  axis <- prcomp(members)$rotation[, 1]
  base <- atan2(axis[2], axis[1]) * 180 / pi
  # prcomp's sign is arbitrary; keep the direction the shoal was laid out in.
  if (abs(((base - shoals$heading[cl] + 180) %% 360) - 180) > 90) base <- base + 180
  base + rnorm(1, 0, 9)
}, numeric(1))
size <- ifelse(is.na(fit$cluster), runif(nrow(pts), 0.55, 0.75), runif(nrow(pts), 0.7, 1.05))
opacity <- ifelse(is.na(fit$cluster), 0.55, 0.92)

fish <- sprintf(
  '<use href="#fish" transform="translate(%.1f %.1f) rotate(%.1f) scale(%.2f)" fill="%s" fill-opacity="%.2f"/>',
  pts$x, pts$y, heading, size, col, opacity
)

svg <- sprintf('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">
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
</svg>', w, h, w, h, hex, hex, sage, paste(fish, collapse = "\n"), cx, cy + 19)

svg_path <- file.path(out, "favicon.svg")
writeLines(svg, svg_path)
rsvg::rsvg_png(svg_path, "man/figures/logo.png", width = w, height = h)

# Favicons: the same artwork at the sizes pkgdown's generator would emit.
sizes <- c(
  "favicon-96x96.png" = 96L, "apple-touch-icon.png" = 180L,
  "web-app-manifest-192x192.png" = 192L, "web-app-manifest-512x512.png" = 512L
)
for (name in names(sizes)) {
  s <- sizes[[name]]
  rsvg::rsvg_png(svg_path, file.path(out, name), width = s, height = round(s * h / w))
}
# The .ico goes through rsvg as well; magick's own SVG rasteriser is coarse.
ico_png <- tempfile(fileext = ".png")
rsvg::rsvg_png(svg_path, ico_png, width = 48L, height = round(48 * h / w))
magick::image_write(magick::image_read(ico_png), file.path(out, "favicon.ico"), format = "ico")
cat("logo and favicons written\n")
