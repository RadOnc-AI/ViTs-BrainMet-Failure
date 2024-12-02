## working directory is the upper directory because main.py will be executed, so import like this
from util.util import init, wandblogger
from util.training_base import TrainingBase, CoxTrainingBase

# all = ["init","TrainingBase","CoxTrainingBase","wandblogger"]