Drop the reference analysis figures here.

Option A - a folder per class:
  raw/Cyclonic Storm/fig1.png
  raw/Severe Cyclonic Storm/fig2.png

Option B - class in the file name (flat folder):
  raw/fig1__Cyclonic Storm.png

Class names must match ml/model.py CLASS_LABELS:
  No Cyclone, Depression, Deep Depression, Cyclonic Storm,
  Severe Cyclonic Storm, Very Severe Cyclonic Storm,
  Extremely Severe Cyclonic Storm

Then run: python backend/scripts/train_and_report.py

