# Disaster Response Pipeline Project

## Table of contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Motivation](#project-motivation)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing-authors-acknowledgements)

## Installation
Install the necessary requirements by running
``` 
pip install -r requirements.txt
```

## Usage

The code consists of three major parts:
1. Data Processing: The data is already uploaded to this repository. If you want to execute the data processing again, you shall perform the following steps:
  *  navigate to the `data` folder.
  * execute `python process_data disaster_messages.csv disaster_categories.csv DisasterResponse.db`
2. Model Identification: Trains a model based on the saved data. For this:
  *  navigate to the `models` folder.
  * execute `python train_classifier ../data/DisasterResponse.db classifier.pkl`
3. Web App: Runs the Flask Web App Server, to display results and classify custom messages. To use the web app:
  *  navigate to the `app` folder.
  * execute `python run`
  * go to https://127.0.0.0:3000 in your browser.

## Project Motivation
This project is a udacity project.
Additional features compared to the task description:
- The dataset is imbalanced. Therefore the Synthetic Minority Over-sampling Technique (SMOTE) is used to oversample the minority classes. You can find the research paper here: [Link](https://www.jair.org/index.php/jair/article/view/10302).
- Testing is done for most of the functions.

## File Descriptions
High level description of the files in this repository.
* `.vscode/`:  
    Contains settings for Visual Studio Code.

* `app/`:  
    Contains the flask web app.

* `data/`:  
    Contains the data preproces.ing pipeline

* `models/`:  
    Contains the training process of the NLP model.

* `preparation/`:  
    Contains jupyter notebooks and other files to play around with in an exploratory style.

* `test_data/`:  
    Contains the data necessary for testing.

* `tests.py`:  
    Contains the tests executed on the code.

* `test.ps1`:  
    This is a powershell script for running pytest and flake8.


## Licensing, Authors, Acknowledgements
Thanks to Figure Eight/Appen to provide the dataset for this Udacity project.

Otherwise, feel free to use the code here as you would like! 