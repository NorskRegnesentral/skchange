.. _contributing_code:

=================
Contributing code
=================
After setting up your development environment as described in the
:doc:`setup`, you are ready to start contributing code to Skchange!

Working on an issue
-------------------
When working on an issue, you should create a new branch from the ``main`` branch in
your local, forked repository. The branch name should be descriptive of the work you
are doing, e.g., ``fix-issue-123`` or ``add-feature-xyz``.

When writing git commit messages, we encourage the use of `Conventional Commits
<https://www.conventionalcommits.org>`_.

Remember that every commit is checked by ``pre-commit`` if setup as described in the
:doc:`setup`. The same checks are also run as part of the continuous integration (CI)
pipeline on GitHub Actions when you push your branch to your forked repository and
create a pull request.

To run the test suite locally, run ``pytest`` in the terminal or use the built-in
`testing <https://code.visualstudio.com/docs/python/testing>`_ features of VS Code.

Pull requests
-------------
When you have completed your work on an issue, you
`create a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_
to the ``main`` branch of the Skchange repository.
Here are some guidelines to make a pull request:

* Title:
* Description:
* Make sure your branch is :ref:`up to date <updating_development_environment>`
  with the latest changes from the ``main`` branch of the Skchange repository.
* Make sure all continuous integration checks pass.
