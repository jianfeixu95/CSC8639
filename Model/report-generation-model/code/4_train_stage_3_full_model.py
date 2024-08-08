from src.full_model.train_full_model import TrainFullModel
import yaml
import os
from utils import utils


def run_train_stage_3_full_model(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    classification_models = TrainFullModel(
        **config['train']['full_model'],
        **config['runs'],
        **config['dataset'],
        **config['model']
    )
    classification_models.train()

    checkpoints_path = os.path.join(
        config["runs"]["path_runs_full_model"],
        "run_{}".format(config["train"]['full_model']['run']),
        "checkpoints"
    )
    best_trained_model_name = os.path.basename(utils.get_best_trained_model_path(checkpoints_path))
    config['evaluation']['name_of_best_full_model_checkpoint'] = best_trained_model_name
    config['generation']['path_to_best_full_model_weights'] = os.path.join(checkpoints_path,best_trained_model_name)

    with open(path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)


if __name__ == '__main__':
    config_path = 'Model/report-generation-model/code/config.yaml'
    run_train_stage_3_full_model(config_path)
