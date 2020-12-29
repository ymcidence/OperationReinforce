from time import gmtime, strftime
import os
from meta import ROOT_PATH


def prepare_training(task_name):
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(ROOT_PATH, 'result', task_name)
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return result_path, save_path, summary_path
