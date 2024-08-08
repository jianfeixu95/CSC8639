from src.full_model.generate_reports_for_images import GeneratedReportsForImages
import yaml


def run_generate_reports(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    generate_reports = GeneratedReportsForImages(**config['generation'])
    generate_reports.generate()


if __name__ == '__main__':
    config_path = "Model/report-generation-model/code/config.yaml"
    run_generate_reports(config_path)
