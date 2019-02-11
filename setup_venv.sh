#!/usr/bin/env bash

virtualenv --python=python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
