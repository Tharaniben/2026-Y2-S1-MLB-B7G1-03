#!/bin/bash
# Run this once to initialise git and push to GitHub
# Replace YOUR_GITHUB_USERNAME and YOUR_REPO_NAME before running

if [ -f "mushrooms.csv" ]; then
    mv mushrooms.csv data/raw/mushrooms.csv
    echo "Moved mushrooms.csv to data/raw/"
fi

git init
git add .
git commit -m "Initial commit — project structure, 6 independent preprocessing notebooks, dataset"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

# Create all model branches from main
git checkout -b model/logistic-regression && git push origin model/logistic-regression && git checkout main
git checkout -b model/decision-tree       && git push origin model/decision-tree       && git checkout main
git checkout -b model/random-forest       && git push origin model/random-forest       && git checkout main
git checkout -b model/svm                 && git push origin model/svm                 && git checkout main
git checkout -b model/knn                 && git push origin model/knn                 && git checkout main
git checkout -b model/neural-network      && git push origin model/neural-network      && git checkout main
echo "All 6 model branches created and pushed. Share branch names with members."
