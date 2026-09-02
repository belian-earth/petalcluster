# Default cluster palette

Qualitative colours for up to eight clusters, a continuous rainbow
beyond that. Up to eight, the ColorBrewer "Dark 2" palette: distinct,
saturated hues that hold up on white and print well. Past eight no
qualitative set stays distinguishable, so the turbo colour map is
sampled evenly instead; adjacent cluster IDs then get adjacent hues,
which is at least legible.

## Usage

``` r
shoal_palette(n)
```

## Arguments

- n:

  Number of colours.

## Value

A character vector of `n` hex colours.

## Examples

``` r
shoal_palette(3)
#> [1] "#1B9E77" "#D95F02" "#7570B3"
shoal_palette(12)
#>  [1] "#30123BFF" "#4454C4FF" "#4490FEFF" "#1FC8DEFF" "#29EFA2FF" "#7DFF56FF"
#>  [7] "#C1F334FF" "#F1CA3AFF" "#FE922AFF" "#EA4F0DFF" "#BE2102FF" "#7A0403FF"
```
