.. _for_maintainers:

===============
For maintainers
===============

Release flow
------------
Skchange is published to
`Github <https://github.com/NorskRegnesentral/skchange/releases>`_
and `PyPI <https://pypi.org/project/skchange/>`_.
New releases are marked by a git tag.

`Release Drafter <https://github.com/release-drafter/release-drafter>`_
is used to automatically create Github releases with release notes based on merged
pull requests.

To make a release, follow these steps:

1. Decide on a `new version <https://semver.org/spec/v2.0.0-rc.2.html>`_.
2. Make sure your fork is up to date with the `main` branch of the main Skchange repository.
3. Create a new branch, e.g. `release-x.y.z`.
4. Bump the version in two files:

   * ``pyproject.toml``: ``version`` field.
   * ``skchange/__init__.py``: ``__version__`` variable.

5. Commit and push the version changes.
6. Open a pull request to the main branch of the Skchange repository with the title: ``release: Skchange <version-number>``.
7. Wait for the GitHub Actions checks to pass, then squash and merge your pull request.
8. Go to the `release <https://github.com/NorskRegnesentral/skchange/actions/workflows/release.yaml>`_
   workflow, click "Run workflow" in the top right,
   select the workflow inputs in the drop-down menu,
   and finally click the "Run workflow" button below the selected inputs.
9. Wait for the workflow to finish, then check
   `Github <https://github.com/NorskRegnesentral/skchange/releases>`_
   and `PyPI <https://pypi.org/project/skchange/>`_.
   to verify that the release was created successfully.
