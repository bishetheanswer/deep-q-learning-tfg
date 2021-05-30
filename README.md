<img src="https://www.esiiab.uclm.es/TallerProgramacionYRoboticaESII/Code.org_files/logomarca%201%20UCLM%20color.jpg" alt="drawing" width="150"/>

# Deep Reinforcement Learning applied to retro videogames

## Training

## Evaluation
Inside the evaluation folder you can find the following files:
* `eval.py`: code for evaluating DQN agents.
* `random_eval.py`: code for evaluating random agents.
* `model.py`: the implementation of the DQN architecture.
* `wrappers_eval.py`: wrappers applied to the environment when evaluating.

You can also find a folder containing the best DQN agent for each game. To see these agents in action you need to execute them using `eval.py`. The videos of their progress during training can be seen here:
* Columns: https://youtu.be/BWTVoRRe5KQ
* Flicky: https://youtu.be/qyLJ7IhasuE
* Bio-Hazard Battle: https://youtu.be/1ZIdxuhISDM
* Streets of Rage 2: https://youtu.be/_7QPvCW4j3g
* Sonic The Hedgehog: https://youtu.be/N2wQk5ypmA8
* Sonic The Hedgehog (rings): https://youtu.be/cKtCdReYTqg
* Sonic The Hedgehog (xpos): https://youtu.be/o6dYv10j_9E

Here is a video of the best agents playing:
* Best agents: https://youtu.be/FoARRAapR_Y

## Tools


<img src="http://www.securizame.com/wp-content/uploads/2016/05/Python-logo-notext.svg_.png" alt="drawing" width="70"/><img src="https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/pytorch-logo-dark.png" alt="drawing" width="250"/> <img src="http://static1.squarespace.com/static/5e6be21d9b6785669d860a72/t/5fc3bb423c6ccf69f3d061a4/1606662982674/openai-logo-horizontal-gradient.jpg?format=1500w" width="150"/> <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" width="150"/>  <img src="https://img.icons8.com/color/452/google-cloud.png" width="100"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/1200px-Amazon_Web_Services_Logo.svg.png" width="100"/>  

* Python 3.7.9
* PyTorch 1.6.0
* Gym Retro 0.8.0
* Kaggle was used at first to perform an informal search in order to select the final hyperparameters for training. The downside is that you are only able to execute a notebook for 9 hours straight so the final training could not be performed using Kaggle.
* I used the Notebooks API from Google Cloud as an alternative to Kaggle but the notebooks stopped their execution after 24 hours.
* The final training was performed in a laboratory of the Escuela Superior de Ingenieria Informática in Albacete. In order to receive the results of the training in my computer I used the S3 (Scalable Storage in cloud) service from AWS. I modified the traning algorithm in order to upload the results to the S3 AWS service.

