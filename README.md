# Rumos Bank Marketing Campaign

#### Objective

This project focuses on predicting whether customers will subscribe to a term deposit, aiming to optimize the bank's marketing strategy and reduce costs. Using a supervised machine learning approach, we analyzed customer data, engineered features, and built multiple models to accurately predict customer behavior. We then tracked experiments using MLFlow, and deployed the best model using FastAPI, ensuring efficient deployment and continuous monitoring.

**Assumption:** In this scenario, where the total baseline cost of false positives (€3.67MM) outweighs that of false negatives (€1.78MM), prioritizing the reduction of false positives becomes imperative for minimizing the overall financial burden on the business. Therefore, the predictive model was optimized for sensitivity (recall).

![Models Compare](https://github.com/PolinaBurova/Supervised-ML-SavingsAccount-Prediction/blob/main/ML_Models_Compare.png)

The best-performing model, Logistic Regression, was selected for its ability to minimize the bank's losses while ensuring effective cost management. By implementing this model, the project successfully reduced potential costs by approximately half a million euros (30%), demonstrating its practical value in a real-world financial setting.


![Threshold Cost](https://github.com/PolinaBurova/Supervised-ML-SavingsAccount-Prediction/blob/main/Cost_Threshold.png)

*This graph shows the relationship between the cost and threshold for the Logistic Regression model. The model was fine-tuned to select the optimal threshold (0.7), which minimized the cost to €1.29 million. This represents a significant reduction in potential losses (baseline cost at €1.79 million), underlining the model's effectiveness in cost management.*


![PRC Curve](https://github.com/PolinaBurova/Supervised-ML-SavingsAccount-Prediction/blob/main/PRC_Curve.png)

*The goal is to predict sensitivity: correctly identifying positive instances in an imbalanced dataset. The Precision-Recall Curve above illustrates the trade-off between precision and recall for the Logistic Regression model. With an Area Under the Curve (AUC) of 0.37, this model was optimized to strike a balance between precision and recall, prioritizing the minimization of false positives and ensuring more accurate customer targeting.*

#### Model Optimization
**Model Validation:** We utilized Cross-validation (K-Fold) to ensure a more accurate and reliable validation process. This approach helps to maximize the use of our data for both training and validation purposes, thereby providing a better assessment of the model's performance.

**Feature Selection:** Feature selection was not explicitly performed in this process because Principal Component Analysis (PCA) was employed. PCA serves to reduce the dimensionality of the dataset, effectively transforming the features into a set of linearly uncorrelated components. This step was also excluded for time-saving purposes. Recursive Feature Elimination could've been used if we hadn't used PCA.

**Hyperparameter Tuning:** To optimize the logistic regression model, we performed hyperparameter tuning using Grid Search Cross Validation. This method was selected for its simplicity and effectiveness in identifying the best hyperparameters. The hyperparameter grid focused on tuning the regularization parameter 𝐶, which controls the trade-off between achieving a low error on training data and minimizing the norm of the weights, thereby preventing overfitting. Grid Search with a cross-validation of 5 folds was used to identify the optimal C value.


**The general notebook containing all the analysis and machine learning algorithms, titled "Supervised-ML-BankPrediction-Classification," is located in the notebooks folder.**

## Project Structure

- `data/`: Contains the dataset file.
- `notebooks/`: Contains the main Jupyter notebook for the project.
- `requirements.txt/`: Lists the necessary dependencies to set up the environment.
- `.vscode/`: Contains the YAML file for activating the environment.
- `notebooks_mlflow/`: Stores the notebooks used for MLFlow tracking.
- `mlruns/`: Directory for MLruns and models. (not in the repository, see below*)
- `MlFlow Screenshots/`: Contains screenshots of various MLFlow runs and models.
- `src/`: Includes the code to run FastAPI.
- `tests/`: Contains the notebook with test requests and prediction screenshots.



## Install the environment

To install the environment with all the necessary dependencies, we should run the following commands:

1. Open a terminal

2. Create a new environment using Python=3.11 with the following command:
    ```
    conda create -n rumos_bank python=3.11
    ```
3. Activate the newly created environment
     ```
    conda activate rumos_bank
    ```
4. Install the required libraries with the following command:
    ```
    conda install pandas numpy scikit-learn
    ```
5. Install additional libraries using pip (from requirements.txt):
     ```
    pip install ipykernel waitress fastapi uvicorn requests pytest mlflow
    ```
6. Install iPython kernel associated with "rumos_bank" environment:
    ```
    python -m ipykernel install --user --name rumos_bank --display-name "rumos_bank"
    ```

If we want to recreate the environment later, you can export the environment configuration to a YAML file with the following commands:

```
conda env export --no-builds --file conda.yml
conda deactivate
conda env remove --name rumos_bank
conda env list  # (Optional: Check if the environment was removed)
conda env create -f conda.yml
conda activate rumos_bank
```


This way, users can easily recreate the environment using the provided conda environment file.


## Visualize runs through UI (MlFlow): 

To keep it more clear and organized, each ML model has been split into separate notebooks which already include the below steps, located in /notebooks_mlflow. Each notebook has the necessary pre-processing steps and MLFlow configurations. This will ensure that each notebook can run independently without missing any required steps.

1.  Define the path where MLflow will store tracking data:
    ```
    uri = Path('C:\\...\\mlruns\\')
    uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri_as_uri)
    ```

2. Define MLFlow Experiment (e.g. Logistic Regression)
    ```
    mlflow.set_experiment("Logistic_Regression_Experiment")
    ```

3. Define the pipeline for pre-processing steps.

3. Start a new run; Log the parameters and end with logging the final model. E.g. Logistic Regression run:
    
- Log parameters, metrics, and model artifacts with MLflow:
    ```
    mlflow.log_params(clf_lr.best_params_)
    mlflow.log_metric("accuracy", score)
    mlflow.log_metric("total_cost", cost)
    mlflow.log_metric("min_cost_threshold", min_threshold[0])
    ```

- Log the final model:
    ```
    mlflow.sklearn.log_model(clf_lr.best_estimator_, artifact_path="logistic_regression_pipeline", registered_model_name="logistic_regression_pipeline", input_example=X_train)
    ```

4. End the run:
    ```
    mlflow.end_run()
    ```

5. View the run in MLFlow UI - in terminal, activate rumos_bank environment:
    ```
    conda activate rumos_bank
    ```
6. Run the following to activate the MLFlow UI:
    ```
    mlflow ui --backend-store-uri file:///C:/.../mlruns --port 5050
    ```
7. Wait for http://127.0.0.1:5050 to be generated and open it in browser. This will show UI of MLFlow with all current runs, their registered models with latest versions. The metrics used are minimum cost generated by the model, threshold cost at 0.5, and accuracy, as seen in original notebooks. The screenshots of the views of each model are saved in MlFlow Screenshots.


## Testing the registered models

1. To test the registered models, go to notebook "mlflow_read_models" in notebooks_mlflow. There we can predict the output of each model and its version. Currently, it's set to logistic_regression_pipeline version 2, which is the latest version of this model: 
    ```
    model_name = "logistic_regression_pipeline"
    model_version = "2"

    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
    ```

3. Then, we created a sample of our dataset and ran a prediction using the model:
    ```
    model.predict(input_data.drop("y", axis=1))
    ```
    

## Running the application

1. Before running the application, go to config/app and fill in the details with the best model tested. In this case, Logistic Regression, and its latest version, 3:
    ```
    "model_name": "logistic_regression_pipeline",
    "model_version": 3,
    "tracking_uri": "C:\\Users\\...\\mlruns"
    ```

2. Go to src/app.py, activate the current environment (rumos_bank) and run the command:
    ```
    `python src/app.py`
    ```

This application will read the config/app.json with the best model, which will be loaded into the application.
It will generate a code http://127.0.0.1:5003 which will open FastAPI application (screenshots saved in /tests).


## Testing the application

1. While the http://127.0.0.1:5003 is still running, go to tests/test_requests
2. Load "requests" and generate a list of samples from the dataset.
3. Run the code 
    ```
    response = requests.post("http://127.0.0.1:5003/predict", json=request_dict)"
    ```
    it should give a prediction of the outcome based on the input data.


##  *Note on Deletion of mlruns Folder
Please note that the mlruns folder has been deleted from this repository. The primary reason for this decision is the repository size: The size of the mlruns folder was too large, making it impractical to include in the repository, especially for version control with GitHub.

If you need to track experiments and runs, please set up a local instance of MLflow and ensure that your environment is properly configured to handle the logging and storage of experiment data. Here are the steps to do so:

1. Install MLflow: Ensure MLflow is installed in your environment.
 ```
pip install mlflow
 ```

2. Set Up MLflow Tracking URI: Configure the tracking URI to a location that suits your needs (local filesystem, remote server, etc.).
```
import mlflow
mlflow.set_tracking_uri('your_tracking_uri')
```

3. Run Your Experiments: Execute your experiments as usual, ensuring that they log to the configured tracking URI.
