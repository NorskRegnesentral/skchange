---
title: "skchange: Fast and flexible change detection in time series"
tags:
  - Python
  - time series
  - changepoint detection
  - anomaly detection
  - machine learning
  - statistics
authors:
  - name: Martin Tveten
    corresponding: true
    affiliation: "1"
  - name: Johannes Voll Kolstø
    affiliation: "1"
  - name: Per August Jarval Moen
    affiliation: "2"
affiliations:
  - index: 1
    name: Norwegian Computing Center, Norway
  - index: 2
    name: Department of Mathematics, University of Oslo, Norway
date: 29 September 2026
bibliography: paper.bib
---

# Summary

Skchange is a Python package for finding structural changes in time series.
It supports two closely related tasks: changepoint detection, where the goal is
to locate times at which the statistical behavior of a sequence changes, and
segment anomaly detection, where the goal is to identify unusual contiguous
intervals. These problems arise in settings such as industrial monitoring,
environmental data analysis, fraud detection, and scientific experiments where
a sequence must be monitored for regime changes or unusual events
[@tveten2022scalable; @gong2025changepoint; @rousseeuw2019robust].

The package provides a uniform, scikit-learn-like interface for fitting
detectors, producing changepoint or anomaly predictions, calibrating penalties,
and evaluating results. It includes exact and approximate search procedures,
cost-based methods, and methods based on statistical tests. Implementations are
written to be efficient on modern Python workflows, with optional numba
acceleration for the main computational bottlenecks.

# Statement of need

Recent research on offline change detection has produced a large collection of
fast and statistically grounded algorithms, but many of these methods are still
hard to use in practice from Python. Available code is often tied to a single
paper, narrowly scoped to one data model, or does not expose a stable interface
 that makes methods easy to compare, extend, and integrate into larger analysis
pipelines.

Skchange was developed to close that gap for researchers and practitioners who
need maintainable, reusable implementations of modern change detection methods.
Its target users include statisticians working on new methodology, applied
researchers analyzing time-indexed data, and machine learning practitioners who
want change detection components that fit naturally into Python-based workflows.

The package is designed around three practical needs. First, users need access
to modern algorithms beyond classical cost-minimization tools, including
methods based on statistical tests that are useful in high-dimensional settings.
Second, users need a common interface across changepoint detection, segment
anomaly detection, scoring, calibration, and evaluation so that workflows do
not have to be rebuilt for every method. Third, contributors need extension
points that make it possible to add new interval scorers, penalties, and search
procedures without rewriting the surrounding infrastructure.

# State of the field

The closest Python package in this area is ruptures, which provides several
widely used algorithms for offline changepoint detection and a broad collection
of cost functions [@truong2020selective]. Ruptures is an important reference
point, but its center of gravity is cost-based segmentation. In contrast,
skchange is built to support both cost-based and test-based detectors under a
common interface. This matters because several recent methods with strong
statistical guarantees, especially for high-dimensional data, are more naturally
expressed through test statistics than through penalized costs.

Broader time-series frameworks such as sktime and aeon provide general
infrastructure for time-series learning, but they are not focused on offering a
specialized collection of modern offline change detection algorithms
[@loning2019sktime; @middlehurst2024aeon]. Skchange instead focuses on this
methodological niche and on efficient implementations of the core primitives
that dominate run time in change detection.

The package was developed as a separate project rather than as a contribution to
an existing library because it combines several requirements that are not jointly
served elsewhere: support for both changepoint and segment anomaly detection,
support for both sparse and dense changes in multivariate data, tools for
automatic false-alarm calibration, and a composable architecture built around
search algorithms, interval scorers, and penalties.

# Software design

Skchange follows scikit-learn estimator conventions where possible, using
predictable `fit`, `predict`, and `fit_predict` workflows familiar to Python
users [@pedregosa2011scikit]. The key design decision is to treat a detector as
a composition of three parts: a search strategy, an interval scorer, and a
penalty. This separates statistical modeling choices from optimization and model
selection choices, which makes the library easier both to use and to extend.

The interval scorer abstraction is central to this design. It covers both cost
functions and test statistics, allowing the same detector framework to support
methods such as PELT, CROPS, MOSUM, Seeded Binary Segmentation, CAPA, and
Circular Binary Segmentation [@killick2012optimal; @haynes2017crops;
@eichinger2018mosum; @meier2021mosum; @kovacs2023seeded; @fisch2022capa;
@fisch2022mvcapa; @olshen2004circular]. In practice, this means that new
research ideas can often be expressed by implementing only one new scorer or one
new search procedure, instead of building an entire package around a single
method.

Performance is addressed by vectorizing scorer evaluation over many candidate
intervals and by precomputing reusable sufficient statistics. These choices
reduce repeated work in the core inner loops of detection algorithms. Optional
numba compilation is then used to accelerate these vectorized kernels while
keeping the public interface in pure Python. This balances accessibility for
users with the execution speed required for large simulation studies and applied
time-series workloads.

# Research impact statement

Skchange is intended as reusable research infrastructure rather than a single
analysis script. The package consolidates implementations of methods drawn from
the recent statistical literature, including algorithms for sparse and dense
high-dimensional changes such as ESAC and SUBSET [@moen2024esac;
@tickle2021computationally], and makes them available through a common, tested
Python interface. This lowers the barrier to reproducing published methodology,
benchmarking alternative detectors, and applying modern change detection tools
to new scientific data.

The software is distributed as an installable Python package, documented with a
public documentation site, and developed with automated tests and continuous
integration. It supports current Python versions across major operating systems
and includes utilities for simulation, calibration, plotting, and evaluation in
addition to the core detectors. These features make skchange suitable for both
methodological research and applied analysis, and they position the package as a
maintainable platform for further work on change detection in the Python
ecosystem.

# AI usage disclosure

Generative AI assistance was used in the preparation of this manuscript.
Specifically, GitHub Copilot using GPT-5.4 was used to help translate an earlier
LaTeX manuscript into the JOSS Markdown format and to assist with copy-editing.
The human authors reviewed, edited, and validated the resulting text and remain
responsible for all technical claims, design decisions, citations, and final
wording.

# Acknowledgements

The authors thank the maintainers and contributors of the open-source Python
scientific software ecosystem, especially the communities around scikit-learn,
sktime, aeon, and numba, on which this project builds.

# References
