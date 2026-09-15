#!/bin/bash
# Usage: bash push_update.sh "Added KNN model notebook"
MESSAGE=$1
git add .
git commit -m "${MESSAGE:-Update}"
git push origin main
