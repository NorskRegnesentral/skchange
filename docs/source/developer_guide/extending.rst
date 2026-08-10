.. _extending_skchange:

===================
Extending Skchange
===================
Skchange is designed to be extended. New interval scorers, detectors, and other
components can be added by subclassing the appropriate base class and following
the conventions described here.

Custom interval scorers
-----------------------
The starting point for a new interval scorer is the extension template shipped
with the repository:

    `extension_templates/interval_scorer.py
    <https://github.com/NorskRegnesentral/skchange/blob/main/extension_templates/interval_scorer.py>`_

The template covers all four scorer types (cost, change score, saving, transient
score) and walks through every method you may need to implement, with ``todo``
markers highlighting the parts to fill in. Follow the instructions at the top
of the file; it is the single source of truth for how a scorer should be
written.

For background on the four scorer types and their interval spec shapes, see the
:doc:`Interval scorers section of the concepts page </user_guide/concepts>`.

Custom detectors
----------------
Work in progress.
