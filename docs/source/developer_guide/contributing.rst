.. _contributing_code:

=================
Contributing code
=================
After setting up your development environment as described in the
:doc:`setup`, you are ready to start contributing code to Skchange!

Working on an issue
-------------------
When working on an issue, create a new branch off of ``main`` in your local fork.
The branch name should be descriptive of the work you are doing, e.g.,
``fix-issue-123`` or ``add-feature-xyz``.

When writing git commit messages, we encourage the use of
`Conventional Commits <https://www.conventionalcommits.org>`_.

Every commit is checked by ``pre-commit`` if you set it up as described in
:doc:`setup`. The same checks are also run as part of the continuous integration
(CI) pipeline on GitHub Actions when you push your branch and open a pull
request.

Running tests
-------------
Skchange uses `pytest <https://docs.pytest.org/>`_. Tests live next to the
code they cover, under a ``tests/`` subpackage (e.g.
``skchange/detectors/tests/``).

To run the test suite locally, run ``pytest`` in the terminal or use the built-in
`testing <https://code.visualstudio.com/docs/python/testing>`_ features of VS Code.

Pull requests
-------------
When you have completed your work on an issue, you
`create a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_
to the ``main`` branch of the Skchange repository.
Here are some guidelines to make a pull request:

* Title:
    - Start your pull request title with a
      `conventional commit <https://www.conventionalcommits.org>`_ tag.
      This helps us add your contribution to the right section of the changelog.
      We use the `Angular convention
      <https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type>`_.
      Specifying the scope of the commit is optional.
    - Use a descriptive title starting with an uppercase letter.
      This text ends up in the changelog, so make sure it is meaningful to the
      user. Use single backticks to annotate code snippets. Use active language
      and do not end your title with punctuation.
    - Example: ``feat: Add new cost class GammaCost``.
* Description:
    - `Link <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue>`_
      to the issue you are working on.
    - Add any relevant context that will help the maintainers review your code:
      design decisions, references, benchmarks, screenshots, etc.
* Make sure your branch is :ref:`up to date <updating_development_environment>`
  with the latest changes from the ``main`` branch of the Skchange repository.
* Make sure all continuous integration checks pass.

After you have opened your pull request, a maintainer will review it and
possibly leave comments. Once all issues are resolved, the maintainer will
merge your pull request, and your work will be part of the next Skchange
release.

Keep in mind that your work does not have to be perfect right away. If you are
stuck or unsure about your solution, feel free to open a draft pull request
and ask for help.

Contributing to the documentation
---------------------------------
We highly value properly documented code.
`Sphinx <https://www.sphinx-doc.org/en/master/>`_ and
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ are used to
generate the documentation from the docstrings in the codebase. Please follow
the `numpydoc style guide
<https://numpydoc.readthedocs.io/en/latest/format.html>`_. See
:doc:`coding_guidelines` for the conventions specific to Skchange.

Contributions to the :doc:`user guide <../user_guide/index>` are also welcome!
Raise an issue if you find something missing or unclear.

To build and check the documentation locally, first install the documentation
dependencies:

.. code-block:: bash

    uv pip install -e .[docs]

Then build the documentation by running ``make docs``. The resulting HTML is
written to ``docs/build/html/index.html``.

The documentation is also built as part of the continuous integration pipeline
when you push changes to a branch with an open pull request.

Credits
-------
This page is inspired by the
`scikit-learn contributing guide
<https://scikit-learn.org/stable/developers/contributing.html>`_ and the
`Polars contributing guide
<https://docs.pola.rs/development/contributing/#pull-requests>`_.
