.. _contributing:

============
Contributing
============
After setting up your development environment as described in the
:doc:`setup`, you are ready to start contributing to Skchange!
This guide provides best practices for contributing code, reporting issues,
and submitting pull requests.

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

Pull requests
-------------
