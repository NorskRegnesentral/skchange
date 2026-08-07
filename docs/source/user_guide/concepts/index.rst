.. _concepts:

========
Concepts
========

In Skchange a **change detector** is composed of an **interval scorer** and a
**penalty**. A change detector is the object you use for detecting changes, an
interval scorer is the user-specified component that tells the detector *what
distributional feature of the data to look for changes in*, while the penalty
controls the number of detected events.

This section introduces each of these concepts to give you a high-level
understanding of the library's design.

.. toctree::
    :maxdepth: 2

    change_detectors
    interval_scorers
    penalties
