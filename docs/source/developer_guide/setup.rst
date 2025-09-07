.. _development_setup:

Development setup
=================
This guide provides step-by-step instructions to set up your local development
environment for contributing to Skchange.

Configuring Git
---------------
You need to have a GitHub account and ``git`` installed on your machine.

1.  `Fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ the `Skchange <https://github.com/NorskRegnesentral/skchange>`_ repository.
2.  Clone the forked repository to your local machine and move into it:

    .. code-block:: bash

        git clone https://github.com/<username>/skchange.git
        cd skchange

3.  Set the ``upstream`` remote branch to the original repository to be able to sync your fork later:

    .. code-block:: bash

        git remote add upstream https://github.com/NorskRegnesentral/skchange.git
        git fetch upstream

Installing dependencies
-----------------------
First, you need to have Python installed.
We recommend using the latest Python version.
Make sure to deactivate any active virtual environment before proceeding,
as the following steps will create a new
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ for Skchange.

1.  Create a virtual environment in your skchange project directory.
    We recommend using `uv <https://docs.astral.sh/uv/>`_ with the ``pip`` interface.

    .. code-block:: bash

        pip install uv
        uv venv

    This creates a virtual environment in a folder named ``.venv``.


2.  Activate the virtual environment.

    On Windows:

    .. code-block:: bash

        .venv\Scripts\activate

    On macOS/Linux:

    .. code-block:: bash

        source .venv/bin/activate

3.  Install an editable version of Skchange with all development dependencies:

    .. code-block:: bash

        uv pip install -e .[dev]

Code quality checks
-------------------
We use `pre-commit <https://pre-commit.com/>`_ for automatic code quality checks.
Pre-commit is installed as part of the development dependencies.

To enable pre-commit hooks, run the following command:

.. code-block:: bash

    pre-commit install

At this point, you should be ready to start contributing to Skchange!

Visual Studio Code configuration
--------------------------------
Using Visual Studio Code (VS Code) with the right extensions can significantly enhance
your development experience. Here is a recommended configuration.

1. Install `VS Code <https://code.visualstudio.com/>`_.
2. Install the following extensions:

   - Python (ms-python.python): Official Python extension for easily working with Python code.
   - Ruff (charliermarsh.ruff): A fast Python linter and code formatter.
     This will help you adhere to the code style requirements enforced by pre-commit.

3. Make sure to select the created virtual environment as the `interpreter <https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment>`_ in VS Code.

.. _updating_development_environment:

Updating the development environment
------------------------------------
It is good practice to regularly update your development environment to ensure you have
the latest version of Skchange and its dependencies.
If not, you may encounter issues when trying to merge your changes with the main codebase
and running the continuous integration tests.

First, sync your fork with the upstream repository:

.. code-block:: bash

    git fetch upstream
    git checkout main
    git merge upstream/main

Then update your dependencies to the latest allowed versions:

.. code-block:: bash

    uv pip install -e .[dev] --upgrade
