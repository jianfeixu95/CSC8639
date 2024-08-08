from src.full_model.test_set_evaluation import EvaluateFullModel
import yaml


def run_test_full_model(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    evaluate_full_model = EvaluateFullModel(
        **config['evaluation'],
        **config['dataset'],
        **config['runs'],
        **config['model']
    )
    evaluate_full_model.evaluate()


if __name__ == '__main__':
    config_path = 'Model/report-generation-model/code/config.yaml'
    run_test_full_model(config_path)
