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

The source folderis divided as:
- `main_train.py`: the main training of all the models.
- `preprocess/`: functions and algorithms to preprocess the text (principally for the keywords extraction).
- `train/`: all the functions and algorithm to prepare the journals embeddings.
- `evaluate/`: all the functions and algorithm to evaluate the journals embeddings.
- `utils/`: some utils like Input/Output class.

# Deployment

The deployed algorithm is SBERT because it is tested algorithm with the best performance. Unfortunaly, it is also the slowest, therefore in case the application will need a fast response time, probably the best option will be the TFIDF.

In order to build and deploy the services you should have installed `docker` and `docker-compose` and run: 
```
docker-compose build
docker-compose -d up
```

You will acess the web application at: [http://172.18.0.3:8501/](http://172.18.0.3:8501/)

