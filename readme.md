# Repo
Kristof created a repo learning-driver-preferences
Folders data, documentation, models, notebooks, scripts
Data will not be included in git-travel (in .gitignore)
Kristof invited teammates Karen and Maxime

# Setup
Install Conda or Anaconda
Clone the repository that Kristof has made: https://github.com/kristof-opero/learning-driver-preferences.git
Create a conda environment driver-preferences with the environment.yml file and activate

conda env create -f environment.yml
conda activate driver-preferences

# Updates
If other packages are needed, please update environment.yml & add, commit and push to the main branch.
To (de)install packages from the updated file: conda env update -f environment.yml --prune

# Update with notebook collaboration helpers
Added nbstripout, nbdime, pre-commit in environment.yml
- nbstripout → strips notebook outputs on commit
- nbdime → improves notebook diffs and merges
- pre-commit → enforces hooks automatically

If installed yet, update your conda environment.

Run following commands in command line:
nbstripout --install
nbdime config-git --enable
pre-commit install
