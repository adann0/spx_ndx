.PHONY: all fetch plot test clean consensus stresstest plotmodel

all: fetch plots

fetch:
	python3 spx_ndx/scrapers/fetch_all.py

plot:
	python3 spx_ndx/plots/generate_all.py

dataset:
	python3 scripts/merge_parquet.py
	python3 scripts/make_dataset.py

test:
	NUMBA_DISABLE_JIT=1 python3 -m pytest tests/ -n auto -v --cov=spx_ndx --cov-report=term-missing

clean:
	rm -rf datas/*.parquet
	rm -rf output/*.png

consensus:
	python3 -m spx_ndx.models.spx_consensus spx_consensus.yaml

stresstest:
	python3 -m spx_ndx.models.spx_consensus.stresstest

plotmodel:
	python3 -m spx_ndx.models.spx_consensus.plot
