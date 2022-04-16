# Frontiers

This documentation shows how to train and deploy the Frontiers Journals Recommendation System. The documentation is divided in *Training* and *Deployment*.

The report with the final results and inights of the project is available at `docs/Report.html`. In the folder `notebooks/` there are the jupyter notebook that creates the report and other drafts of the work.

The main folders of the project are:
- `data`: where all the data are stored during the training phase and used when the model is deployed.
- `docs`: there are the report and the description of the assignment.
- `notebooks`: there are notebook drafts used to design the solution.
- `requirements`: there are the requirements for training and deploy.
- `src`: source code for the training phase.
- `api`: api code for the deployment.
- `app`: front-end code for the deployment.

During the training phase of all the models, I used a CentOS server machine with 40 cores, 30 Gb RAM, and a Nvidia Tesla V100 (with 16Gb RAM). With such configuration the training of all the models and the evaluation lasted about 1 hour. The code should run with both cpu and gpu, altought I do not guarantee a short running time (not tested). Certainly, the pipeline could be reduced training some models, but at the moment it can only be hard coded commenting some part of the code.

The deployment is tested in a normal laptop and it is deployed using docker compose. No gpu is needed for the deployment.

## Training

In order to train the models, the first step is initializing the enviroment:
```
conda create --name frontiers python=3.8 -y
conda activate frontiers
pip install -r requirements/requirements.txt
```

Then the entire pipeline is executed with:
```
python src/main_train.py
```
As previosly introduced, the training of all the pipeline is both time and resources expensive. In order to reduce time, the user has two choices: use gpu or running only some part of the pipeline exluding some models. 

The source folder is divided as:
- `main_train.py`: the main function to train all the models.
- `preprocess/`: functions and algorithms to preprocess the text (principally for the keywords extraction).
- `train/`: all the functions and algorithm to prepare the journals embeddings.
- `evaluate/`: all the functions and algorithm to evaluate the journals embeddings.
- `utils/`: some utils like Input/Output class.

# Deployment

The deployed algorithm is SBERT. During the training phase SBERT performs better than all the other algorithms. Unfortunaly, it is also the slowest. In case the application will need a fast response time, probably the best option will be the TFIDF.

In order to build and deploy the services you should have installed `docker` and `docker-compose` and run: 
```
docker-compose build
docker-compose -d up
```

You acess the web application at: [http://172.18.0.3:8501/](http://172.18.0.3:8501/)

