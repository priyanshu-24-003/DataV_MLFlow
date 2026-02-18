# DataV_MLFlow
DataV_MLFlow is a repository is aimed to perform data Versioning using MLFlow on Forked Repo: palmer-penguins-classification.


## src

        src contains components of pipeline

## experiments

        -contains original ipynb notebook created by: https://github.com/theodoretnguyen
        - orignal repo : https://github.com/theodoretnguyen/palmer-penguins-classification
        - my forked Mlops Version : https://github.com/priyanshu-24-003/palmer-penguins-classification

## data 

        contains followings:

                source csv
                
                artifacts:
                    processed data
                    models 
                    reports
                    logs for different components
                
                    * though not pushed to remote repo.
## plumber
        
        plumber handles pipeline as all the other plumbers do :)

        run on local machine : python plumber.py

## Notes

        1. once you make the plumber run the pipeline expect following artifacts:

                    /data/logs
                    /data/interim
                    /data/raw
                    /data/models
                    /data/processed
                    /data/reports

        
        2. see artifacts related to different experiments that i ran using:

                        mlflow ui: https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow

## miscllaneous tasks achieved:
        
        1. ran the whole pipeline using mlflow :
                using modular/functional programming 
                used context manager of mlflow.start_run().

        2. saved currect_exp.log that "
                stores the log of all components in current experiment
                mlflow tracks it , logged data/currect_exp.log
                and file becomes empty again to be used in next experiment

        
        3. saved artifacts directories from plumber.

                        logged data/processed
                        logged data/models
                        logged data/reports
                        logged src/

        4. pushed repo to dagshub with remote branch in git
                1. saving versioned data
                        - pushed data/raw
                                data/interim
                                data/processed 
                                etc
                                to dagshub s3://
                                dagshub upload --bucket priyanshu24003/DataV_MLFlow ./data  data/

                2. remote tracking server

                        -created seperate branch DagsHub(Now Default) for remote tracking in git.
                       
                        -Experiment tracking :
                                I used DagsHub and MLflow here
                       
                        -pushed repo to dagshub.

        5. Ran nested runs :

                        trained basic models for later comparision
                        
                        logged metrics of respective basic models 

                        compared accuracies of main model to base models



                        
