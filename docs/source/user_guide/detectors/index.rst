.. _detectors_guide:

=========
Detectors
=========
This section is a per-detector reference within the user guide. Each page
introduces one detector, walks through a small example, and highlights the
parameters that matter most in practice.

Skchange currently ships six change detectors. They all accept an interval
scorer through a typed constructor argument, but differ in which score types
they support and in how they search for changepoints.

Which detector to reach for
---------------------------
The table below compares the detectors at a glance. Column meanings:

- **Scorer**: The interval scorer type the detector accepts through its
  constructor.
- **Few CPs** / **Many CPs**: Run-time on a series of length :math:`n` when the
  true number of changepoints is small vs. proportional to :math:`n`. Reported
  as a qualitative label plus the corresponding big-O complexity.
- **Sparse MV**: Supports array-valued penalties and aggregation schemes tailored
  to sparse changes in multivariate data, where only a few features change.
- **Anomalies**: Naturally produces segment anomalies (pairs of changepoints
  around an anomalous interval against a baseline) rather than a full
  segmentation.
- **Tuning**: How much hyperparameter tweaking is typically needed to get
  reliable performance. "low" means only a penalty or nothing to set,
  "medium" means a couple of additional knobs that matter in practice,
  "high" means a critical hyperparameter that must be chosen carefully.

Follow the detector link for a detailed discussion of the trade-offs.

.. list-table::
    :header-rows: 1
    :widths: 24 16 16 9 9 12

    * - Detector
      - Few CPs
      - Many CPs
      - Sparse MV
      - Anomalies
      - Tuning
    * - :doc:`PELT <pelt>`
      - slow, :math:`O(n^2)`
      - fast, :math:`O(n)`
      - no
      - no
      - low
    * - :doc:`FPOP <fpop>`
      - fast,
      - fast,
      - no
      - no
      - low
    * - :doc:`CROPS <crops>`
      - slow, :math:`O(n^2 \log n)`
      - medium, :math:`O(n \log n)`
      - no
      - no
      - low
    * - :doc:`SeededBinarySegmentation <seeded_binseg>`
      - fast, :math:`O(n \log n)`
      - fast, :math:`O(n \log n)`
      - yes
      - no
      - medium
    * - :doc:`MovingWindow <moving_window>`
      - fast, :math:`O(n)`
      - fast, :math:`O(n)`
      - yes
      - no
      - high
    * - :doc:`CircularBinarySegmentation <circular_binseg>`
      - slow, :math:`O(n^2 \log n)`
      - slow, :math:`O(n^2 \log n)`
      - yes
      - yes
      - medium
    * - :doc:`CAPA <capa>`
      - slow, :math:`O(n^2)`
      - fast, :math:`O(n)`
      - yes
      - yes
      - medium

.. toctree::
    :maxdepth: 1
    :hidden:

    pelt
    fpop
    crops
    seeded_binseg
    moving_window
    circular_binseg
    capa
