# deepscribe
Computer vision experiments with OCHRE cuneiform data. 

# Setup

Install required packages using `conda env create -f environment.yml`

# Data Source

Labeled images originate from the Persepolis Fortification Archive project, consisting of over 100,000 annotated regions of 6064 tablets written in Elamite Cuneiform. 

# Modules

The majority of the `deepscribe` code base consists of Luigi workflow tasks selecting and preprocessing subsets of the image dataset, splitting them into train/test/validation sets, training ML models on the resulting images, and performing model analysis tasks. 

# Documentation

Docs are a work in progress, but can be found at [https://deepscribe.readthedocs.io/en/latest/](https://deepscribe.readthedocs.io/en/latest/)