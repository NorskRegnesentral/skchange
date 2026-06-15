.. _api_reference:

=============
API reference
=============

Welcome to the API reference for ``skchange``.

The API reference provides a technical manual.
It describes the classes and functions included in Skchange.

For Python Interactive examples, see the
`interactive <https://github.com/NorskRegnesentral/skchange/tree/main/interactive>`_
folder on GitHub.

.. note::
    The API documented here lives under ``skchange.new_api.*`` and is the
    target API for the upcoming 0.17.0 release. When 0.17.0 ships, the same
    names will move to the top level (``skchange.detectors``,
    ``skchange.interval_scorers``, ``skchange.penalties``, ...) and the
    ``new_api`` prefix will be dropped. See the
    `migration guide
    <https://github.com/NorskRegnesentral/skchange/blob/main/skchange/new_api/MIGRATION_GUIDE.md>`_
    for details.

.. toctree::
   :maxdepth: 1

   datasets
   detectors
   interval_scorers
   metrics
   penalties
   tuning
   utils
