# A hex sticker in the house style of a5R: black hexagon, sage border,
# a light thin wordmark. The motif is a shoal: a few loose clusters of
# small points, in the package palette, with scattered noise between them.
out <- Sys.getenv("OUT", "pkgdown/favicon")
sage <- "#74ac90"
pal <- c("#1B9E77", "#7570B3", "#E7298A", "#E6AB02")
set.seed(7)
w <- 240; h <- 277
cx <- w / 2; cy <- h / 2
# Pointy-top hexagon matching a5R's 240x277 proportions.
r <- h / 2 - 3
ang <- seq(90, 390, by = 60) * pi / 180
hx <- cx + r * cos(ang) * (w / 2 - 3) / (r * cos(30 * pi / 180))
hy <- cy - r * sin(ang)
hex <- paste(sprintf("%.1f,%.1f", hx, hy), collapse = " ")

centres <- rbind(c(70, 78), c(170, 92), c(85, 205), c(178, 200))
spread <- c(17, 14, 15, 13)
pts <- character()
for (j in seq_len(nrow(centres))) {
  n <- 34
  a <- runif(n, 0, 2 * pi); d <- abs(rnorm(n, 0, spread[j]))
  x <- centres[j, 1] + d * cos(a) * 1.35; y <- centres[j, 2] + d * sin(a)
  rad <- runif(n, 1.6, 2.8)
  op <- 0.45 + 0.5 * exp(-d / spread[j])
  pts <- c(pts, sprintf('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" fill-opacity="%.2f"/>', x, y, rad, pal[j], op))
}
# Sparse noise between the shoals.
nx <- runif(40, 30, w - 30); ny <- runif(40, 40, h - 40)
keep <- apply(centres, 1, function(cc) sqrt((nx - cc[1])^2 / 1.8 + (ny - cc[2])^2)) |> apply(1, min) > 24
pts <- c(pts, sprintf('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#9aa5a0" fill-opacity="0.45"/>', nx[keep], ny[keep], runif(sum(keep), 1.2, 1.9)))

svg <- sprintf('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">
<polygon points="%s" fill="#000000" stroke="%s" stroke-width="2.5" stroke-linejoin="round"/>
<clipPath id="hex"><polygon points="%s"/></clipPath>
<g clip-path="url(#hex)">
%s
</g>
<text x="%.1f" y="%.1f" text-anchor="middle" font-family="Inter, Lato, Helvetica, Arial, sans-serif" font-weight="200" font-size="54" fill="#dfe6e2" letter-spacing="1.5">shoal</text>
</svg>', w, h, w, h, hex, sage, hex, paste(pts, collapse = "\n"), cx, cy + 19)
writeLines(svg, file.path(out, "logo.svg"))
rsvg::rsvg_png(file.path(out, "logo.svg"), file.path(out, "logo.png"), width = w, height = h)
# Favicons: the same artwork at the sizes pkgdown's generator would emit.
for (s in c(96L, 180L, 192L, 512L)) {
  name <- switch(as.character(s), "96" = "favicon-96x96.png", "180" = "apple-touch-icon.png",
                 "192" = "web-app-manifest-192x192.png", "512" = "web-app-manifest-512x512.png")
  rsvg::rsvg_png(file.path(out, "logo.svg"), file.path(out, name), width = s, height = round(s * h / w))
}
ico <- magick::image_read(file.path(out, "logo.svg"), density = 300)
magick::image_write(magick::image_scale(ico, "48x48"), file.path(out, "favicon.ico"), format = "ico")
file.copy(file.path(out, "logo.svg"), file.path(out, "favicon.svg"), overwrite = TRUE)
cat("logo written\n")
