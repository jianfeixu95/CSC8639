import os
import yaml
from utils import utils
from src.object_detector.training_script_object_detector import TrainObjectDetector


def run_train_stage_1_object_detector(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    object_detector = TrainObjectDetector(config)
    object_detector.train()

    checkpoints_path = os.path.join(
        config["runs"]["path_runs_object_detector"],
        "run_{}".format(config["train"]['object_detector']['run']),
        "weights"
    )
    best_trained_model_path = utils.get_best_trained_model_path(checkpoints_path)
    config['train']['classification_modules']['path_to_best_object_detector_weights'] = best_trained_model_path
    config['train']['full_model']['path_to_best_object_detector_weights'] = best_trained_model_path
    config['evaluation']['path_to_best_object_detector_weights'] = best_trained_model_path
    config['generation']['path_to_best_object_detector_weights'] = best_trained_model_path
    with open(path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)


if __name__ == '__main__':
    config_path = 'Model/report-generation-model/code/config.yaml'
    run_train_stage_1_object_detector(config_path)
