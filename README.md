# Frontiers

This documentation shows how to train and deploy the Frontiers Journals Recommendation System. The documentation is divided in *Training*, and *Deployment*.

The report with the final results and inights of the project is available at `docs/Report.html`. In the folder `notebooks/` there are the jupyter notebook that creates the report and other draft of the work.

The principal folders of the projects:
- `data`: where all the data are stored during the training phase and used when the model is deployed.
- `docs`: there are the report and the description of the assignment.
- `notebooks`: there are notebook draft used to think the solution.
- `requirements`: there are the requirements for training and deploy.
- `src`: source code for the training phase.
- `api`: api code for the deployment.
- `app`: front-end code for the deployment.

During the training phase of all the models I used a CentOS server machine with 40 cores and a Nvidia Tesla V100 (with 16Gb RAM). With such configuration the training of all the models and the evaluation lasted about 1 hour. The code should run with both cpu and gpu, altought I do not guarantee the running time. Certainly, the pipeline could be reduced only trained some models, but at the moment it should be hard coded commenting some part of it.

The deployment is tested in a normal laptop and it is deployed using docker compose. No gpu is needed for the deployment.

## Training

In order to train the models, the first is to initialize the enviroment:
```
conda create --name frontiers python=3.8 -y
conda activate frontiers
pip install -r requirements/requirements.txt
```



## Using docker

# Deployment

```
docker run -v /home/operti/inda/frontiers/data/:/app/data/ -p 8501:8501 app_frontiers
```

