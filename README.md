# DataV_MLFlow
DataV_MLFlow is a repository is aimed to perform data Versioning using MLFlow on Forked Repo: palmer-penguins-classification.

## src

        src contains components of pipeline

## data 

        contains followings:

                source csv
                
                artifacts:
                    raw data
                    interim data
                    processed data
                    models 
                    reports

                logs for different components

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

                    <-------coming soon ------>


## miscllaneous tasks achieved:

        1. ran the whole pipeline using mlflow :
                using modular/functional programming and directly using 

        2. saved currect_exp.log that "
                stores the log of current experiment
                mlflow tracks it 
                and file becomes empty again to be used in next experiment
        
        3. saved artifacts as whole two dirs from plumber 

                        logged data/
                        logged src/

        4. pushed repo to dags with remote branch in git
                1. saving versioned data
                        - pushed data/raw
                                data/interim
                                data/processed 
                                etc
                                to dagshub s3://
                                dagshub upload --bucket priyanshu24003/DataV_MLFlow ./data  data/

                2. remote tracking server

                        -created seperate branch DagsHub(Now Default    ) for remote work
                        
                        -pushed repo to dagshub and id experiment tracking usually as we learned

        5. Ran nested runs :

                        trained basic models for later comparision
                        
                        logged metrics of respective basic models 

                        compared accuracies of main model to base models



                        
