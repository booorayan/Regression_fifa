# Using Machine Learning Regression Models to Predict the Outcome of Football Matches
#### The project aims to prdict the number of goals by a home and away team to help a betting company come up with odds for various matches. It also seeks to predict the outcome of a footbal match between team 1 and team 2. 

#### The dataset used for this project was sourced from https://drive.google.com/file/d/1BYUqaEEnFtAe5lvzJh9lpVpR2MAvERUc/view

#### 17/08/2019

## Libraries/Technologies
The libraries required for this project included:

      pandas - for performing data analysis and cleaning.

      numpy - used for fast matrix operations.

      matplotlib - used to create plots.

      seaborn - used for creating plots.  
      
      sklearn - machine learning library      
  
The language used was python3 and the regression models uesd for the project were  logistic and polynomial regression models. 

## Description
The objective of the project was to predict the number of goals scored by the home team, goals scored by the away team and the match outcome between team 1 and team 2.

To achieve these objectives, data cleaning, EDA and Modelling were conducted. 
For modelling, polynomial and logistic models were employed. In total, four models were created, two polynomial models, one logistic model and one xgboost classifier model.

To challenge the solution, xgboost classifier model was used. 

### Experiment Design
This project followed the CRISP-DM methodology for the experiment design. The CRISP-DM methodology has the following steps:

####   1.   Problem understanding: 
Entailed gaining an understanding of the research problem and the objectives to be met for the project. External research was conducted to gain an understanding of the fifa ranking system. 
The metrics for success were also defined in this phase.
Some metrics for success included:
  *   Two polynomial regression models that predict the number of goals scored by the Home and away teams
  *   A logistic regression model that predicts the outcome of a football match between two teams.  
   
####   2.   Data understanding: 
Entailed the initial familiarization with and exploration of the dataset, as well as the evaluation of the quality of the dataset provided for the study.

               
####   3.   Data preparation: 
Involved data cleaning/tidying the dataframe to remove missing values and ensure uniformity of data. 
   
          
####   4.   Modelling: 
Involved the processes of selecting a model technique, selecting the features and labels for the model, generating a test design, building the model and evaluating the performance of the model. 
   
In total, seven models were used for this project, most of which were regression models. The other models were logistic regression model and xgboost model for challenging the solution for logistic regression.

Most of the models exhibited an accuracy ranging between 50-60%. Therefore, one conclusion made is that these models are not the most appropriate for predciting the number of goals scored by two teams.


####   5.   Evaluation: 
XGBoost classification model was used to challenge the predictions by polynomial and logistic regression models. 

The distribution of predictions (i.e. match outcomes) was also analysed and compared with the distribution for the test set.

            
### Conclusion


*   Regression models (polynomial and logistic models) are not the best models for predicting goals scored and match outcomes.
*   An assumption taken for this project is that the ranking system was the same between 2006-2018
*   The accuracy of predictions can be improved by stacking more models together


### License

Copyright (c) 2019 **Booorayan**
