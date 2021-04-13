# Disaster Response Pipeline Project

#### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

### Installation <a name="installation"></a>
- Download and install a recent version of Python Anaconda distribution 
- Clone the repository to a local drive and run all files
- The dataset is available in the repos.

### Project Motivation <a name="motivation"></a>

After a disaster, agencies managing the receive millions of messages from different media or channels. The challenge is usually to filter important messages
and take timely actions, by sending the right messages to appropriate teams or agencies. 
In this project, we will use Natural language procesing (NLP) to classify different messages, develop and deploy a model that can help categorizing different 
incoming messages in case of disaster.

## Project Structure
 ```
 The project has two main files, here is the description:
   Data-Lake-with-Apache-Spark
    |
    |   data
    |      | disaster_categories.csv
    |      | disaster_messages.csv
    |      | ETL Pipeline Preparation.ipynb
    |      | process_data.py    
    |      | 
    |    models
    |      | ML Pipeline Preparation.ipynb
    |      | train_classifier.py
    |   
    |   README.md
 ``` 
### File Description <a name="files"></a>
- `data`: folder has the cvs files used in this analysis
    -  `ETL Pipeline Preparation.ipynb`: This notebook is the prototype of the ETL pipeline for the project. All steps are built incrementally.
    -  `process_data.py ` : This file wraps all steps define in the prototype above to build the entire pipeline.
    -  `disaster_categories.csv` : All messages categories already define
    - `disaster_messages.csv`: All messages used in training and test
- `models`:
    - ML Pipeline Preparation.ipynb: This file is the prototype for the ML pipeline. All steps are built incrementally
    - train_classifier.py : This file wraps all steps define in the ML prototype above to build the entire pipeline.

## Installation 

- Install [python 3.8](https://www.python.org)
- Clone the current repository. 
- Create a python virtual environment 
- Activate the python virtual environment
- Install all package necessary for the project :  `pip install -r requirements.txt`
- From command line, follow the steps below :
    1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    2. Run the following command in the app's directory to run your web app.
        `python run.py`

    3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements<a name="licensing"></a>

You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/search?q=boston+airbnb+in%3Adatasets).  