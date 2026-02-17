"""
plumber handles pipeline
"""
#import dependencies

import logging
import os
import mlflow 

#importing different components
import src.A_data_injestion as A
import src.B_preperation as B
import src.C_feature_selection as C
import src.D_model_training as D
import src.E_model_evaluation as E


#setting mlflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Ensure the "logs" directory exists
log_dir = 'data/logs'

os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('E_model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'E_model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def pipeline():
    
    # Mention your experiment below
    mlflow.set_experiment('BasicTester')

    with mlflow.start_run() as runs:

        print("Running Data_Injestion Component ")
        A.main()
        print()

        print("Running Data_Preparation Component B")
        B.main()
        print()

        print("Running Feature_Selection Component C")
        C.main()
        print()

        print("Running Model_Training Component D")
        D.main()
        print()

        print("Running Model_Evaluation Component E")
        E.main()
        print()

        #Saving the entire data of this run/ sub-experiment
        mlflow.log_artifact('./data/processed')
        mlflow.log_artifact('./data/models')
        mlflow.log_artifact('./data/reports')        


        #Saving the Entire source code realated to this run/ sub-experiment
        mlflow.log_artifact('./src')

        #setting a tag 
        mlflow.set_tags({'ran-by':"rohan", 'test_size':"0.1"})

        mlflow.log_artifact('data/current_exp.log')
        with open('data/current_exp.log', 'w') as f3:
            pass
        
       
if __name__ == "__main__":

    pipeline()