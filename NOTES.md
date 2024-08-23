# Intended users

1. Myself and close collaborators.
2. Other statistics and ML researchers, data scientists and engineers.

Primarly for research and development, but also viable for production use at some point.

# Wanted features

## Overall principles

1. Computational performance: numba.
2. Extendability: Implement non-restrictive base classes, such that new types of algorithms can easily be implemented without being forced into a template it doesn't fit.
3. Composition: Easy to combine algorithms. Pipelining, ensembling, etc.
4. Interoperability: Easy to use with other libraries. For example, it is great if
meta algorithms like pipelines from other libraries can be used with skchange.

Allways keep in mind

1. Scikit-learn design principles https://arxiv.org/pdf/1309.0238.pdf

2. sktime design: https://arxiv.org/abs/2101.04938



## Data

1. Univariate time series: A single variable over time.
2. Multiple time series: A collection of unrelated univariate time series.
3. Multivariate time series: A collection of related univariate time series.

Use pd.DataFrame for all types of input data.

## Anomaly detectors
Finds point or collective anomalies in data.
In general, the .predict method returns positions of anomalies in the data.
One entry or row in the output should correspond to one anomalous event.
Scores can be returned by .predict_scores

The different output formats for .predict for different tasks and data is listed below.
The examples use integer indices or intervals, but both floats and datetimes are supported interchangably.

### Example data
```python
import pandas as pd

x_univariate = pd.Series([0, 2, 0, 0, 0, 1, 1, 0, 0, 0])
x_multivariate = pd.DataFrame(
    [
        [0, 0],
        [2, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 0]
    ],
    columns = ["a", "b"]
)
```
```python
x_univariate
0    0
1    2
2    0
3    0
4    0
5    1
6    1
7    0
8    0
9    0
dtype: int64
```
```python
x_multivariate
   a  b
0  0  0
1  2  0
2  0  0
3  0  0
4  0  0
5  0  1
6  0  1
7  0  0
8  0  0
9  0  0
```

### Point anomalies in univariate or multivariate data without subset anomalies
```python
detector = PointAnomalyDetector().fit(x_univariate)
detector.predict(x_univariate)
0    1
1    5
2    6
dtype: int64
# same output for x_multivariate
```
### Subset point anomalies in multivariate data
```python
detector = PointAnomalyDetector().fit(x_multivariate)
detector.predict(x_multivariate)
   index columns
0      1     [0]
1      5     [1]
2      6     [1]
dtypes: int64, list
```
### Collective anomalies in univariate or multivariate data without subset anomalies
```python
detector = CollectiveAnomalyDetector().fit(x_univariate)
detector.predict(x_univariate)
0    [1, 1]
1    [5, 6]
dtype: interval  # closed on both ends.
# same output for x_multivariate
```
### Subset collective anomalies in multivariate data
```python
detector = CollectiveAnomalyDetector().fit(x_multivariate)
detector.predict(x_multivariate)
   index columns
0  [1, 1]     [0]
1  [5, 6]     [1]
dtypes: interval, list
```


## Change detectors
Segments data into homogenous segments.
The .predict method returns positions of changepoints in the data, where a changepoint is defined as the last index in a segment.
One entry or row in the output should correspond to one change event.
Scores can be returned by .predict_scores

The different output formats for .predict for different tasks and data is listed below,
using the same example data as for anomaly detection.

### Changepoints in univariate data or multivariate data without subset changes
```python
detector = ChangeDetector().fit(x_univariate)
detector.predict(x_univariate)
0    0
1    1
2    4
2    6
dtype: int64
# same output for x_multivariate
```
### Subset changepoints in multivariate data
```python
detector = SubsetChangeDetector().fit(x_multivariate)
detector.predict(x_multivariate)
   index columns
0      0     [0]
1      1     [0]
2      4     [1]
3      6     [1]
dtypes: int64, list
```


## General detectors requirements and features
1. Both testing/scoring based, cost based, or anything else.
2. Various types of thresholds/penalties.
3. Subset anomalies.
3. Possibility to tune number of detections in a general way, across both change and anomaly detectors.
4. Should be possible to add specialised tuning procedures for each algorithm or subclasses:

    a. Add option for .tune() method per algorithm. Plays poorly with pipelines and composition in general (?).

    b. Add specialised tuning classes with a detector as a component (like CV in sklearn and sktime) than can only be used based on (i) a tag or (ii) inheritance.

    c. Add a `tune_penalty` = True/False or `penalty` = `tune`, which governs what happens in .fit().

6. Add .show() method to visualise results?
7. Option to implement quick updating of fit and predict with new data, without having to retrain the entire model. But fallback on retraining the entire model.


## Nice-to-have features

1. A wrapper for turning av detector for univariate time series to a detector for multiple time series.
2. An aggregator that aggregates scores from multiple detectors into a single score.
3. Make a wrapper for diagnosing anomalies post detection? Or pipeline step?
4. Ability to set up a pipeline of model -> drift adaptation -> anomaly detection


# Why depend on sktime?

1. Tidy interface.
2. Clear purpose of base class.
3. Clarity of design principles.

    a. https://arxiv.org/pdf/1309.0238.pdf

    b. https://arxiv.org/abs/2101.04938

4. Well documented.
5. BaseAnnotator is a non-restrictive base class for change detection algorithms. Avoids the need to implement a lot of boilerplate code.
6. Several useful meta algorithms are already implemented in sktime. For example, pipelines.


# Related packages

- sktime: https://www.sktime.net/en/stable/
- darts: https://unit8co.github.io/darts/
- nixtla: https://github.com/Nixtla

See https://www.sktime.net/en/stable/related_software.html for a more complete list.

# Development workflow

1. Create a new algorithm by using the annotator extension template of sktime: https://github.com/sktime/sktime/blob/df21a0c0275ebf28deb30efac5d469c9f0d178e3/extension_templates/annotation.py.
2. Explore the new algorithm class, its method and its component functions in an interactive script. These scripts are located in the `interactive` folder and named `explore_<algorithm>.py`, for example `explore_pelt.py`, if the algorithm is `pelt`.
In the future, these explorative scripts might be run as part of the CI/CD pipeline.
3. Write pytests in the relevant folders' `tests` subfolder. If the algorithm is named `pelt` and located in `skchange/change_detection/pelt.py`, write tests in `skchange/change_detection/tests/test_pelt.py`.


More resources: https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Coding standards: https://github.com/sktime/sktime/blob/df21a0c0275ebf28deb30efac5d469c9f0d178e3/docs/source/developer_guide/coding_standards.rst#L65

# Release workflow
For releasing a new version of skchange, run the do-nothing script `build_tools/make_release.py` for instructions. See the script for more information.

# Roadmap

- ~~Implement PELT as first test implementation of an algorithm.~~
- ~~Implement seeded binary segmentation with CUSUM as a second test implementation of an algorithm.~~
- ~~Implement CAPA.~~
- ~~Complete first version of README.~~
- ~~Complete the make_release do-nothing script.~~  Also?? https://stackoverflow.com/questions/72270892/git-versioning-with-setuptools-in-pyproject-toml
- ~~Publish to PyPI? When this is done, add to make_release script.~~
- Add automatic documentation generation by Sphinx and readthedocs: https://eikonomega.medium.com/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365.
Get access to skchange from readthedocs. Add documentation generation to make_release script.
