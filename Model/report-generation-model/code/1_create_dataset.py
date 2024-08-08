from src.dataset.create_dataset import CreateDataset
import yaml


def run_create_dataset(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    create_dataset = CreateDataset(**config['dataset'], **config['logger'])
    create_dataset.create()


if __name__ == '__main__':
    config_path = "Model/report-generation-model/code/config.yaml"
    run_create_dataset(config_path)
