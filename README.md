# MedEvalKit

**MedEvalKit** is a modular and extensible Python toolkit for evaluating machine learning models, especially in healthcare applications. It provides unified APIs for computing metrics, calibration curves, bootstrap confidence intervals, and plotting diagnostic curves.

## Features

- Binary and multiclass classification support
- Threshold optimization
- Calibration error estimation
- Bootstrap confidence intervals
- ROC, PR, and calibration plotting
- Simulate results at different incidence rates

## Installation

```bash
pip install medevalkit
```

## Usage Example

```python
from medevalkit import Evaluate

clf.fit(X_train, y_train)

evaluator = Evaluate(clf, X_test, y_test, classification=True, threshold=threshold)
report = evaluator.generate_report(bootstrap=True)
print(report['text_report'])
```

## GitHub Repo

Visit https://github.com/wesleyyeung/medevalkit for more examples.

## License

This project is licensed under the MIT License.
