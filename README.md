# Disaster Response Pipeline Project

### Background

During severe disasters, the disaster response oganizations aim at providing necessary salvation and support in time for targets as precisely as possible. Data from various social media serve as an important channel for the organizations to acquire relevant information which can support with decision-making. For example, a community could suffer from food and drinkable water shortage during a hurricane and some people may post on their social media softwares, searching for help: 'We have been trapped here during the hurricane and don't have enough food!' And their can be numerous messages posted on the social media. The response oganizations want to automatically detect the relevant messages and figure out if someone is in trouble and what kind of help they need, through a proper machine-learning model. This project developed just such a model, expecting to help the organizations improve their performance.

### Files

In data folder:

    - disaster_messages.csv Messages collected from social media
    - disaster_categories.csv Category information of the messages
    - process_data.py Python script for data processing
    
 In models folder:
 
    - train_classifier.py Python script for NLP-ML models
    
 In app folder:
 
    - files intended to build a web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
