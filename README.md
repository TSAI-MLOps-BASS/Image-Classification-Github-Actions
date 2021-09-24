# Introduction to MLOps #

**MLOps** is software development practice that aims to manage, deploy, maintain and automate machine learning related artifacts and processes including but not limited to data, code, experiments and models. What makes it different from DevOps is the coupling of Data with Code in the process. Sitting at the intersection of Data Engineering, Machine Learning and DevOps, its principles span across a number of operational verticals such as Version Control, Data Pipelines, Model and Data Validation, CI/CD and Monitoring.  

## Version Control ##
Reproducibility is one of the most key aspects of any software development project. Since machine learning is very iterative and experimental in nature, maintaining the correct version and state of all the associated components is vital for the success of a project. That translates to maintaining the correct version of data, code, parameters and environment throughout. With the right set of tools such as Git, DVC and FastDS it makes the job of version controlling easy and hassle free.

## Data Pipelines ##
Also known as ETL (Extraction, Transform and Loading)/ELT, pipelines aim to operationalize the data engineering part of machine learning development. It brings in reusability and transparency thereby making the repetitive process more efficient and scalable.

## Model and Data Validation ##
The objective of the development boils down to the statistical acceptance criteria of the model performance as well as the data being served and provided for training. For each of the validations, creating varied test cases is necessary however running those manually for every change in continual fashion is not a practical solution. Given the complexity and iterative nature it is a much harder problem to solve and thankfully MLOps is there that addresses it and provides insightful information for timely decision.

## CI/CD ##
Standing for Continuous Integration and Continuous Deployment, it automates the deployment process of the code and data thereby leaving the developers and data scientists to focus more on development and experiments.

## Monitoring ##
With the end goal to consume the trained model in the target application for better outcome, it is always of interest that the model performs at the best and the desired environmental setup is highly available for the model to serve better. It is important to monitor the model health, traffic, latency, errors, logs, saturation etc. Taking feedback in the loop is also a very important aspect to ensure the model performance is optimum. MLOps provides a set of tools and practices that automate this monitoring process that helps in taking the appropriate decision in a timely fashion.

While at a high level we discussed about the four components above, there are many other things involved in MLOps practice that we will discover and share as we make progress.

# Introduction to DVC #
It is a version control tool that works hand in hand with Git. It helps us to manage large size artefacts such as the dataset and models seamlessly while Git takes care of remaining files.

(More to come here...)

# Assignment #
The task is to redo the following tutorial. The model in the tutorial is made with tensorflow keras. However, we need to recreate the model in Pytorch and complete the activities outlined in the tutorial.

https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial


## Team Members ##
- Bhagabat
- Shreeyash
- Aswathi
