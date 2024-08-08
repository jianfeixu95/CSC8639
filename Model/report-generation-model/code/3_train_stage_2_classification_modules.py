from src.full_model.train_full_model import TrainFullModel
import yaml
import os
from utils import utils


def run_train_stage_2_classification_modules(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    classification_models = TrainFullModel(
        **config['train']['classification_modules'],
        **config['dataset'],
        **config['runs'],
        **config['model']
    )
    classification_models.train()

    checkpoints_path = os.path.join(
        config["runs"]["path_runs_full_model"],
        "run_{}".format(config["train"]['classification_modules']['run']),
        "checkpoints"
    )
    best_trained_model_path = utils.get_best_trained_model_path(checkpoints_path)
    config['train']['full_model']['path_to_best_classification_modules_weights'] = best_trained_model_path

    with open(path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)


if __name__ == '__main__':
    config_path = 'Model/report-generation-model/code/config.yaml'
    run_train_stage_2_classification_modules(config_path)
