# Crime Predictor using Environmental Features
This project is used as a capstone project for [Eskwelabs](https://www.eskwelabs.com/). In this work, we developed models used to predict reported crime using different geospatial (light posts, train stations, etc.) and temporal (reported crime for the past week) features using geohashes and a weekly basis.

## Setup
Run 
```s
conda env create -f <environment-name>.yml
```
to create a virtual environment.
You may also use the `requirements.txt` to install the dependencies.

Notable dependencies include:
- numpy=1.16.4
- pandas=0.25.1
- matplotlib=3.1.0
- scikit-learn=0.21.2
- geopandas=0.4.1
- geohash=1.0

## Directory
- **data/**: contains the files where the raw and preprocessed data is stored.
- **notebooks/**: contains the Jupyter notebooks used.
- **pickle/**: contains the pickle files for the models.
- **source/**: contains the class file for CrimePredictor and configuration files.
- **utils/**: contains utility methods like instantiating the model and loading the data.

## Supplementary Slides
- [Capstone Presentation Slides](https://docs.google.com/presentation/d/1xraAIao6-gYqxSH9hs9HzhmVblcnkLuNCT-w9H8vT3E/edit?usp=sharing)
