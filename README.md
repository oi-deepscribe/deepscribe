# deepscribe
Computer vision experiments with OCHRE cuneiform data. 

# Setup

Install required packages using `conda env create -f environment.yml`

# Data Source

Labeled images originate from the Persepolis Fortification Archive project, consisting of over 100,000 annotated regions of 6064 tablets written in Elamite Cuneiform. 

# Modules

The majority of the `deepscribe` code base consists of Luigi workflow tasks selecting and preprocessing subsets of the image dataset, splitting them into train/test/validation sets, training ML models on the resulting images, and performing model analysis tasks. 

## `deepscribe.luigi.preprocessing`

- `OchreToHD5Task`: collects images with filenames in `<sign>-<image uuid>-<object uuid>` format from a provided folder and collects them in a HDF5 file. 

## `deepscribe.luigi.image_processing`

- `ProcessImageTask`: Superclass whose method `process_image` is applied to every matrix in an HDF5 file. 
- `ImagesToGrayscaleTask`: `process_image` converts the image to a single-channel grayscale image.
- `StandardizeImageSizeTask`: Scales and black-pads the image to a square. 
- `RescaleImageValuesTask`: Normalizes each image. 
- `AddGaussianNoiseTask`: Adds gaussian noise to each image as a data augmentation procedure. 

## `deepscribe.luigi.ml_input`

- `SubsampleDatasetTask`: Selects the class labels provided out of the full dataset and saves a hdf archive with the reduced dataset.
- `AssignDatasetTask`: Randomly partitions the data into a train/test/validation split and saves as a .npz archive.

## `deepscribe.luigi.training`

- `TrainKerasModelFromDefinitionTask`: Reads a JSON file containing model architecture parameters and trains the model, validating on validation split. 
- `RunTalosScanTask`: Reads a JSON file containing ranges of architecture parameters, runs a scan with Talos. 
- `TrainSKLModelFromDefinitionTask`: Reads a JSON file containing scikit-learn model parameters, trains model. 

## `deepscribe.luigi.model_selection`
- `TestModelTask`: Evaluates trained model on test data, saves confusion matrix to npz archive.
- `PlotConfusionMatrixTask`: Plots confusion matrix. 
- `GenerateClassificationReportTask`: Generates a classification report using `sklearn`. 
- `PlotIncorrectTask`: Generates a sample of incorrectly classified images with their true labels for model analysis. 
- `RunAnalysisOnTestDataTask`: Wrapper task performing the previous model selection tasks. 

## `deepscribe.models`
 - `cnn_classifier_2conv`: CNN with two convolutional layers, inspired by AlexNet.  