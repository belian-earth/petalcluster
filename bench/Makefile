.PHONY: all install document test bench bench-r bench-sklearn compare \
       gen-data clean-data clean-bench clean

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

# -- Package -------------------------------------------------------------------

install:
	cd $(ROOT) && NOT_CRAN=true R CMD INSTALL .

document:
	cd $(ROOT) && Rscript -e 'rextendr::document()'

test:
	cd $(ROOT) && Rscript -e 'devtools::test()'

# -- Benchmarks ----------------------------------------------------------------

all: compare

gen-data:
	cd $(ROOT) && Rscript bench/gen_data.R

bench-r: gen-data install
	cd $(ROOT) && Rscript bench/bench_r.R

bench-sklearn: gen-data
	cd $(ROOT) && uv run bench/bench_sklearn.py

bench: bench-r bench-sklearn

compare: bench
	cd $(ROOT) && Rscript bench/compare.R

# -- Cleanup -------------------------------------------------------------------

clean-data:
	rm -rf $(ROOT)/bench/data

clean-bench: clean-data
	rm -f $(ROOT)/bench/results_r.csv $(ROOT)/bench/results_sklearn.csv $(ROOT)/bench/scaling.png

clean: clean-bench
