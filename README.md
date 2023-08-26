# shoppingPredictor
A classifier to predict whether or not a user will make a purchase.


## Description

Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — the classifier will predict whether or not the user will make a purchase.

## Install Libraries
```
pip install -r requirements.txt
```

## Example
```
python shopping.py shopping.csv
    Correct: 4098
    Incorrect: 834
    True Positive Rate: 37.92%
    True Negative Rate: 91.55%
```