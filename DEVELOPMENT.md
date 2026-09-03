# Development notes

Build and workflow details for working on shoal. Users need none of this.

## Timing anything: install first

`devtools::load_all()` compiles the Rust code without optimisation, because
pkgbuild sets `DEBUG` while it builds, and `devtools::document()` rewrites
`src/Makevars` the same way. That is fine for correctness but timings from a
`load_all()` session are misleading by an order of magnitude or more. For
anything performance-related install the package first, then work from
`library(shoal)`:

```sh
NOT_CRAN=true R CMD INSTALL .
```

The benchmarks in `bench/` do this for you; see `bench/README.md`.

## extendr bindings

After adding or changing a `#[extendr]` function, regenerate
`R/extendr-wrappers.R` before `devtools::document()`; with the package's
current rextendr configuration that is

```r
rextendr:::make_wrappers("shoal", "shoal", "R/extendr-wrappers.R", use_symbols = TRUE)
```

from a `load_all()` session. Keep the Rust doc comments on those bindings as
plain `//` comments, or roxygen will document the internal wrappers.

## Rust tests

The crate has its own unit tests, run from `src/rust/`:

```sh
cargo test --release
```

The EVoC port lives in `src/rust/evoc-core/`; its parity suite against the
Python reference is under `evoc-port/` and runs with `cargo test --release`
from `evoc-port/parity/`.
