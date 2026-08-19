# Data

Keep source datasets in `data/raw/` and derived, analysis-ready datasets in
`data/processed/`.

Do not overwrite raw data in place. If a notebook or script transforms data,
write the output to `data/processed/` and record the script, seed, and package
environment used to create it.
