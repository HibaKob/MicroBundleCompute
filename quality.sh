#!/bin/bash


# historical context: sibl repo
# pytest --cov=geo/src/ptg  --cov-report term-missing

# current context: autotwin repos, either `atmesh` or `atpixel`
# pytest --cov=atmesh --cov-report term-missing
#pytest -v --cov=atpixel  --cov-report term-missing

pytest -v --cov=microbundlecompute  --cov-report term-missing

