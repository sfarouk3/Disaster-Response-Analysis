
# # Disaster Response Pipeline Project

### Table of Contents

1. [Overview](#Overview)
2. [Installation](#installation)
3. [File Content](#files)
4. [User Guide](#user_guide)
5. [Licensing, Authors, and Acknowledgements](#license)


## Overview

Disaster Response is a Web App project implemented with python using ETL and NLP algorithms targeting analysing the messages received during disasters and classifying it to the corressponding categories to accelerate the response.


## Installation <a name="installation"></a>

You Can use plotly library by running "pip install plotly==4.14.3" command.
 3.*.


## File Content<a name="files"></a>
    
data folder:

    -disaster_messages.csv: Disaster Received Messages.
    -disaster_categories.csv: Disaster Messages Categories.
    -process_data.py: Python File for the ETL pipeline

models folder:

    -train_classifier.py: A Python script to create the ML pipeline model 
    
app folder:

    -run.py: Pythone script to run the app.


## User Guide<a name="user_guide"></a>
    
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model in 'pkl' file below numbers show the model evaluation
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        ![alt text](https://github.com/sfarouk3/Disaster-Response-Analysis/blob/main/images/DisRes3.PNG)
        
 

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    


3. Go to http://0.0.0.0:3001/  
![alt text](https://github.com/sfarouk3/Disaster-Response-Analysis/blob/main/images/Web1.PNG)
![alt text](https://github.com/sfarouk3/Disaster-Response-Analysis/blob/main/images/Web2.PNG)


## Licensing, Authors, Acknowledgements<a name="license"></a>

Thanks to "Figure Eight" for providing the dataset for this project.

