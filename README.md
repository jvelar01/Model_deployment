# Model Deployment with Flask and MLflow for Optimization

This project demonstrates the deployment of a machine learning model using Flask and MLflow for optimization. The model is built and trained in a Jupyter notebook, while the Flask application serves as the deployment environment. MLflow is used to manage and track the experiments, ensuring smooth model deployment and optimization.

## Objective

The goal of this project is to deploy a machine learning model that optimizes the EAF process using Flask for the web application interface and MLflow for managing the model lifecycle.

## Methodology

- **Model Building**: The model is constructed and trained in a Jupyter notebook, where various machine learning techniques are applied for optimization.
- **Flask Application**: The Flask app serves as the interface where users can interact with the model and receive predictions.
- **MLflow**: Used for managing the machine learning model, tracking experiments, and organizing model artifacts. MLflow stores this information in:
  - `mlartifacts`: Directory where the trained models and artifacts are saved.
  - `mlruns`: Directory storing the metadata and experiment details.
- **Pipeline**: The code for the pipeline is located in the `prePortugal` script, which prepares the data and feeds it into the model.
- **App Code**: The deployment logic for the Flask app is contained in the `depModelAs` script, which connects the model to the Flask server.

## Documentation

The deployment process is explained in the `assignment 2 mlflow and flask` file, which provides an overview of how MLflow and Flask interact to deliver a fully deployed model.

## Conclusion

This project illustrates how Flask and MLflow can work together to deploy a machine learning model in a production environment. By utilizing MLflow to track and manage experiments and deploying the model with Flask, we have created a streamlined workflow for model optimization and deployment.
