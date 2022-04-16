# Frontiers

This documentation shows how to train and deploy the Frontiers Journals Recommendation System. The documentation is divided into *Training* and *Deployment*.

The report with the final results and insights of the project is available at `docs/Report.html`. In the folder `notebooks/` there is the jupyter notebook that creates the report and other drafts of the work.

The main folders of the project are:
- `data`: where all the data are stored during the training phase and used when the model is deployed.
- `docs`: there are the report and the description of the assignment.
- `notebooks`: there are notebook drafts used to design the solution.
- `requirements`: there are the requirements for training and deployment.
- `src`: source code for the training phase.
- `api`: API code for the deployment.
- `app`: front-end code for the deployment.

During the training phase of all the models, I used a CentOS server machine with 40 cores, 30 Gb RAM, and an Nvidia Tesla V100 (with 16Gb RAM). With such configuration, the training of all the models and the evaluation lasted less than 1 hour. The code should run with both CPU and GPU, although I do not guarantee a short running time (not tested). Certainly, the pipeline could be reduced by training some models, but at the moment it can only be hard-coded by commenting on some part of the code.

The deployment is tested on a normal laptop and it is deployed using docker-compose. No GPU is needed for the deployment.

## Training

In order to train the models, the first step is initializing the environment:
```
conda create --name frontiers python=3.8 -y
conda activate frontiers
pip install -r requirements/requirements.txt
```

Then the entire pipeline is executed with:
```
python src/main_train.py
```
As previously introduced, the training of all the pipelines is both time and resources expensive. In order to reduce time, the user has two choices: use GPU or run only some part of the pipeline excluding some models. 

The source folder is divided as:
- `main_train.py`: the main function to train all the models.
- `preprocess/`: functions and algorithms to preprocess the text (principally for the keywords extraction).
- `train/`: all the functions and algorithms to prepare the journals embeddings.
- `evaluate/`: all the functions and algorithms to evaluate the journals embeddings.
- `utils/`: some utils like Input/Output class.

# Deployment

The deployed algorithm is SBERT. During the training phase, SBERT performs better than all the other algorithms. Unfortunately, it is also the slowest. In case the application will need a fast response time, probably the best option will be the TFIDF.

In order to build and deploy the services you should have installed `docker` and `docker-compose` and run: 
```
docker-compose build
docker-compose -d up
```

You access the web application at [http://172.18.0.3:8501/](http://172.18.0.3:8501/)

In folder `docs/examples/` there some papers that could be tested for the prediction.
