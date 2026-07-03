.. _change_detection:

================
Change detection
================
This section is a hands-on guide to detecting *changepoints* — abrupt changes
in the distribution of a time series — with Skchange. Each subsection focuses
on a specific *distributional feature* (mean, variance, covariance matrix, linear
slope, ...) and shows how to pick a scorer and a detector for that feature, along with
plots of the results.

The material assumes familiarity with the :doc:`Concepts page </user_guide/concepts>`,
which introduces the change detector API, interval scorers, and penalties.

Which detector to reach for
---------------------------
Skchange currently ships four change detectors. They all accept an interval
scorer through a typed constructor argument, but differ in which score types
they support and in how they search for changepoints.

.. list-table::
    :header-rows: 1
    :widths: 22 15 32 31

    * - Detector
      - Scorer type
      - Pros
      - Cons
    * - ``PELT``
      - ``cost``
      - Exact optimisation of the penalised segmentation objective;
        few parameters to tweak to get a guaranteed optimal segmentation;
        near-linear computational complexity when there are many changepoints.
      - Can be slow for large datasets with few changepoints; doesn't support array
        penalties and arbitrary aggregation schemes to tackle sparse changes in
        multivariate data
    * - ``CROPS``
      - ``cost``
      - Extends ``PELT`` by returning the full solution path across a range
        of penalties. Useful for model selection and for inspecting how the
        segmentation changes with the penalty.
      - Same as ``PELT``, and runs ``PELT`` several times, so noticeably slower than a
        single ``PELT`` fit at a known penalty.
    * - ``SeededBinarySegmentation``
      - ``change_score``
      - Fast, log-linear computational complexity no matter the location and number of
        true changepoints; not restricted to cost-based algorithms; supports array
        penalisation and several aggregation schemes to tackle sparse changes in
        multivariate data.
      - Approximate: More parameters to tweak to be ensured a good solution;
        the seeded grid of intervals may miss very short segments unless
        ``min_subinterval_length`` is reduced.
    * - ``MovingWindow``
      - ``change_score``
      - Very fast, linear computational complexity, if you know only a couple of
        bandwiths are relevant to detect changes in your data;
        supports array penalisation and several aggregation schemes to tackle sparse
        changes in multivariate data; intuitive visual score plot for inspection
      - Sensitive to the window size; poor at detecting closely spaced changes.
    * - ``CircularBinarySegmentation``
      - ``transient_score``
      - Tailored to *transient* or *epidemic* changes: short intervals where the
        distribution differs from the surrounding baseline. Each interval
        contributes a pair of changepoints. Supports array penalisation and several
        aggregation schemes to tackle sparse changes in multivariate data.
      - Slow, log-quadratic computational complexity; not designed for persistent
        regime changes.
    * - ``CAPA``
      - ``saving``
      - Jointly detects collective/segment and point anomalies against a fixed baseline.
        Each detected segment anomaly contributes a pair of changepoints.
        Much faster than ``CircularBinarySegmentation``, linear computational complexity
        if there are many anomalies; supports array penalisation and several aggregation
        schemes to tackle sparse changes in multivariate data.
      - Not suitable for classifcal changepoint detection where each segment is assumed
        to be drawn from a different distribution.

.. toctree::
    :maxdepth: 1

    mean
