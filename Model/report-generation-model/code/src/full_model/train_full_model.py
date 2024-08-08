import shutil
from ast import literal_eval
import logging
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from datetime import datetime
from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.evaluate_full_model.evaluate_model import evaluate_model
from src.full_model.report_generation_model import ReportGenerationModel


class TrainFullModel(object):

    def __init__(
            self,
            run,
            seed,
            pretrain_without_lm_model,
            image_input_size,
            percentage_of_train_set_to_use,
            percentage_of_val_set_to_use,
            batch_size,
            effective_batch_size,
            num_workers,
            epochs,
            lr,
            evaluate_every_k_batches,
            patience_lr_scheduler,
            threshold_lr_scheduler,
            factor_lr_scheduler,
            cooldown_lr_scheduler,
            num_beams,
            max_num_tokens_generate,
            num_batches_of_generated_sentences_to_save_to_file,
            num_batches_of_generated_reports_to_save_to_file,
            num_batches_to_process_for_language_model_evaluation,
            num_images_to_plot,
            bertscore_similarity_threshold,
            weight_object_detector_loss,
            weight_binary_classifier_region_selection_loss,
            weight_binary_classifier_region_abnormal_loss,
            weight_language_model_loss,
            path_to_best_object_detector_weights,
            path_to_best_classification_modules_weights,
            # dataset
            auto_delete_dataset_folder,
            num_rows_to_create_in_new_csv_files,
            path_full_dataset,
            path_mimic_cxr,
            path_mimic_cxr_jpg,
            path_chest_imagenome,
            # runs
            auto_delete_run_folder,
            path_runs_object_detector,
            path_runs_full_model,
            path_chexbert_weights):
        #classification_modules
        self.run = run
        self.seed = seed
        self.pretrain_without_lm_model = pretrain_without_lm_model
        self.image_input_size = image_input_size
        self.percentage_of_train_set_to_use = percentage_of_train_set_to_use
        self.percentage_of_val_set_to_use = percentage_of_val_set_to_use
        self.batch_size = batch_size
        self.effective_batch_size = effective_batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = eval(lr)
        self.evaluate_every_k_batches = evaluate_every_k_batches
        self.patience_lr_scheduler = patience_lr_scheduler
        self.threshold_lr_scheduler = eval(threshold_lr_scheduler)
        self.factor_lr_scheduler = factor_lr_scheduler
        self.cooldown_lr_scheduler = cooldown_lr_scheduler
        self.num_beams = num_beams
        self.max_num_tokens_generate = max_num_tokens_generate
        self.num_batches_of_generated_sentences_to_save_to_file = num_batches_of_generated_sentences_to_save_to_file
        self.num_batches_of_generated_reports_to_save_to_file = num_batches_of_generated_reports_to_save_to_file
        self.num_batches_to_process_for_language_model_evaluation = num_batches_to_process_for_language_model_evaluation
        self.num_images_to_plot = num_images_to_plot
        self.bertscore_similarity_threshold = bertscore_similarity_threshold
        self.weight_object_detector_loss = weight_object_detector_loss
        self.weight_binary_classifier_region_selection_loss = weight_binary_classifier_region_selection_loss
        self.weight_binary_classifier_region_abnormal_loss = weight_binary_classifier_region_abnormal_loss
        self.weight_language_model_loss = weight_language_model_loss
        self.path_to_best_object_detector_weights = path_to_best_object_detector_weights
        self.path_to_best_classification_modules_weights = path_to_best_classification_modules_weights
        self.auto_delete_run_folder = auto_delete_run_folder

        # dataset
        self.auto_delete_dataset_folde=auto_delete_dataset_folder
        self.num_rows_to_create_in_new_csv_files=num_rows_to_create_in_new_csv_files
        self.path_full_dataset = path_full_dataset
        self.path_mimic_cxr=path_mimic_cxr
        self.path_mimic_cxr_jpg=path_mimic_cxr_jpg
        self.path_chest_imagenome=path_chest_imagenome

        # runs
        self.auto_delete_run_folder=auto_delete_run_folder
        self.path_runs_object_detector = path_runs_object_detector
        self.path_runs_full_model = path_runs_full_model

        # model
        self.path_chexbert_weights = path_chexbert_weights

        self.run_start_time = datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
        self.log = logging.getLogger(__name__)
        self.set_seed()

    def set_seed(self):
        # set the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train_model(
            self,
            model,
            train_dl,
            val_dl,
            optimizer,
            scaler,
            lr_scheduler,
            current_epoch,
            epochs,
            overall_steps_taken,
            lowest_val_loss,
            checkpoints_folder_path,
            tokenizer,
            generated_sentences_and_reports_folder_path,
            writer,
            log_file,
    ):
        """
        Train a model on train set and evaluate on validation set.
        Saves best model w.r.t. val loss.

        Parameters
        ----------
        model: nn.Module
            The input model to be trained.
        train_dl: torch.utils.data.Dataloder
            The train dataloader to train on.
        val_dl: torch.utils.data.Dataloder
            The val dataloader to validate on.
        optimizer: Optimizer
            The model's optimizer.
        lr_scheduler: torch.optim.lr_scheduler
            The learning rate scheduler to use.
        epochs: int
            Number of epochs to train for.
        checkpoints_folder_path: str
            Path to folder where checkpoints with best weights will be saved.
        tokenizer: transformers.GPT2Tokenizer
            Used for decoding the generated ids into tokens in evaluate_model (more specifically evaluate_language_model)
        generated_sentences_folder_path:
            Path to folder where generated sentences will be saved as a txt file.
        writer: torch.utils.tensorboard.SummaryWriter
            Writer for logging values to tensorboard.
        log_file: str
            Path to file where error messages will be logged.

        Returns
        -------
        None, but saves model weights every time the val loss has decreased below the last lowest val loss.
        """
        run_params = {}
        run_params["epochs"] = epochs
        run_params["checkpoints_folder_path"] = checkpoints_folder_path
        run_params["lowest_val_loss"] = lowest_val_loss
        run_params["best_epoch"] = None  # the epoch with the lowest val loss overall
        run_params["overall_steps_taken"] = overall_steps_taken  # for logging to tensorboard
        run_params["log_file"] = log_file  # for logging error messages (e.g. OOM)

        # for gradient accumulation
        ACCUMULATION_STEPS = self.effective_batch_size // self.batch_size

        # to recover from out of memory error if a batch has a sequence that is too long
        oom = False

        for epoch in range(current_epoch, epochs):
            run_params["epoch"] = epoch
            self.log.info(f"Training epoch {epoch}!\n")

            train_losses_dict = {
                "total_loss": 0.0,
                "obj_detector_loss": 0.0,
                "region_selection_loss": 0.0,
                "region_abnormal_loss": 0.0,
            }

            if not self.pretrain_without_lm_model:
                train_losses_dict["language_model_loss"] = 0.0

            run_params["steps_taken"] = 0  # to know when to evaluate model during epoch and to normalize losses

            for num_batch, batch in tqdm(enumerate(train_dl)):
                images = batch["images"]
                image_targets = batch["image_targets"]
                region_has_sentence = batch["region_has_sentence"]
                region_is_abnormal = batch["region_is_abnormal"]

                batch_size = images.size(0)

                images = images.to(self.device, non_blocking=True)
                image_targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in image_targets]
                region_has_sentence = region_has_sentence.to(self.device, non_blocking=True)
                region_is_abnormal = region_is_abnormal.to(self.device, non_blocking=True)

                if self.pretrain_without_lm_model:
                    input_ids = None
                    attention_mask = None
                else:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(images, image_targets, input_ids, attention_mask, region_has_sentence,
                                       region_is_abnormal)

                        # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
                        # this can happen if e.g. the object detector did not detect any regions in an image (e.g. there are a couple of lateral chest x-rays in ChestImaGenome,
                        # even though the dataset should only include frontal chest x-rays. These bad input images can trigger output == -1)
                        if output == -1:
                            with open(run_params["log_file"], "a") as f:
                                f.write("Training:\n")
                                f.write(
                                    f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")

                            optimizer.zero_grad()
                            continue

                        if self.pretrain_without_lm_model:
                            (
                                obj_detector_loss_dict,
                                classifier_loss_region_selection,
                                classifier_loss_region_abnormal,
                            ) = output
                        else:
                            (
                                obj_detector_loss_dict,
                                classifier_loss_region_selection,
                                classifier_loss_region_abnormal,
                                language_model_loss,
                            ) = output

                        # sum up all 4 losses from the object detector
                        obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

                        # sum up the rest of the losses
                        total_loss = (
                                self.weight_object_detector_loss * obj_detector_losses + self.weight_binary_classifier_region_selection_loss * classifier_loss_region_selection + self.weight_binary_classifier_region_abnormal_loss * classifier_loss_region_abnormal
                        )

                        if not self.pretrain_without_lm_model:
                            total_loss += self.weight_language_model_loss * language_model_loss

                    scaler.scale(total_loss).backward()

                except RuntimeError as e:  # out of memory error
                    self.log.info(f"Error: {e}")
                    if "out of memory" in str(e):
                        oom = True

                        with open(run_params["log_file"], "a") as f:
                            f.write("Training:\n")
                            f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                            f.write(f"Error message: {str(e)}\n\n")
                    else:
                        raise e

                if oom:
                    # free up memory
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    oom = False
                    continue

                if (num_batch + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                list_of_losses = [
                    total_loss,
                    obj_detector_losses,
                    classifier_loss_region_selection,
                    classifier_loss_region_abnormal,
                ]

                if not self.pretrain_without_lm_model:
                    list_of_losses.append(language_model_loss)

                # dicts are insertion ordered since Python 3.7
                for loss_type, loss in zip(train_losses_dict, list_of_losses):
                    train_losses_dict[loss_type] += loss.item() * batch_size

                run_params["steps_taken"] += 1
                run_params["overall_steps_taken"] += 1

                # evaluate every k batches and at the end of each epoch
                if run_params["steps_taken"] >= self.evaluate_every_k_batches or (num_batch + 1) == len(train_dl):

                    self.log.info(f"Evaluating at step {run_params['overall_steps_taken']}!")
                    evaluate_model(
                        model,
                        train_losses_dict,
                        val_dl,
                        lr_scheduler,
                        optimizer,
                        scaler,
                        writer,
                        tokenizer,
                        run_params,
                        generated_sentences_and_reports_folder_path,
                        self.pretrain_without_lm_model,
                        self.weight_object_detector_loss,
                        self.weight_language_model_loss,
                        self.weight_binary_classifier_region_selection_loss,
                        self.weight_binary_classifier_region_abnormal_loss,
                        self.batch_size,
                        self.num_beams,
                        self.max_num_tokens_generate,
                        self.num_batches_of_generated_sentences_to_save_to_file,
                        self.num_batches_of_generated_reports_to_save_to_file,
                        self.num_batches_to_process_for_language_model_evaluation,
                        self.num_images_to_plot,
                        self.bertscore_similarity_threshold,
                        self.path_chexbert_weights
                    )
                    self.log.info(f"Metrics evaluated at step {run_params['overall_steps_taken']}!")

                    # set the model back to training
                    model.train()

                    # reset values for the next evaluation
                    for loss_type in train_losses_dict:
                        train_losses_dict[loss_type] = 0.0
                    run_params["steps_taken"] = 0
                    optimizer.zero_grad()

        self.log.info("Finished training!")
        self.log.info(
            f"Lowest overall val loss: {run_params['lowest_val_loss']:.3f} at epoch {run_params['best_epoch']}")
        return None

    def get_model(self, checkpoint=None):
        # if there is a key error when loading checkpoint, try uncommenting down below
        # since depending on the torch version, the state dicts may be different
        # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
        # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")

        model = ReportGenerationModel(
            path_to_best_object_detector_weights=self.path_to_best_object_detector_weights,
            pretrain_without_lm_model=self.pretrain_without_lm_model
        )
        model.to(self.device, non_blocking=True)

        if checkpoint:
            model.load_state_dict(checkpoint["model"])
        model.train()

        return model

    def seed_worker(self, worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data_loaders(self, tokenizer, train_dataset, val_dataset):

        custom_collate_train = CustomCollator(tokenizer=tokenizer, is_val_or_test=False,
                                              pretrain_without_lm_model=self.pretrain_without_lm_model)
        custom_collate_val = CustomCollator(tokenizer=tokenizer, is_val_or_test=True,
                                            pretrain_without_lm_model=self.pretrain_without_lm_model)

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_loader = DataLoader(
            train_dataset,
            collate_fn=custom_collate_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            generator=g,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            collate_fn=custom_collate_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            # could also be set to NUM_WORKERS, but I had some problems with the val loader stopping sometimes when num_workers != 0
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_transforms(self, dataset: str):
        # see compute_mean_std_dataset.py in src/dataset
        mean = 0.471
        std = 0.302

        # use albumentations for Compose and transforms
        # augmentations are applied with prob=0.5
        # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
        train_transforms = A.Compose(
            [
                # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
                # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
                # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
                # INTER_AREA works best for shrinking images
                A.LongestMaxSize(max_size=self.image_input_size, interpolation=cv2.INTER_AREA),
                A.ColorJitter(hue=0.0),
                A.GaussNoise(),
                # randomly (by default prob=0.5) translate and rotate image
                # mode and cval specify that black pixels are used to fill in newly created pixels
                # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
                A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
                # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
                A.PadIfNeeded(min_height=self.image_input_size, min_width=self.image_input_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # don't apply data augmentations to val set (and test set)
        val_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.image_input_size, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=self.image_input_size, min_width=self.image_input_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        if dataset == "train":
            return train_transforms
        else:
            return val_transforms

    def get_tokenized_datasets(self, tokenizer, raw_train_dataset, raw_val_dataset):
        def tokenize_function(example):
            phrases = example["bbox_phrases"]  # List[str]
            bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
            eos_token = "<|endoftext|>"

            phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

            # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
            return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

        tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
        tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

        # tokenized datasets will consist of the columns
        #   - mimic_image_file_path (str)
        #   - bbox_coordinates (List[List[int]])
        #   - bbox_labels (List[int])
        #   - bbox_phrases (List[str])
        #   - input_ids (List[List[int]])
        #   - attention_mask (List[List[int]])
        #   - bbox_phrase_exists (List[bool])
        #   - bbox_is_abnormal (List[bool])
        #
        #   val dataset will have additional column:
        #   - reference_report (str)

        return tokenized_train_dataset, tokenized_val_dataset

    def get_tokenizer(self):
        checkpoint = "healx/gpt-2-pubmed-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def get_datasets(self, config_file_path):
        usecols = [
            "mimic_image_file_path",
            "bbox_coordinates",
            "bbox_labels",
            "bbox_phrases",
            "bbox_phrase_exists",
            "bbox_is_abnormal",
        ]

        # all of the columns below are stored as strings in the csv_file
        # however, as they are actually lists, we apply the literal_eval func to convert them to lists
        converters = {
            "bbox_coordinates": literal_eval,
            "bbox_labels": literal_eval,
            "bbox_phrases": literal_eval,
            "bbox_phrase_exists": literal_eval,
            "bbox_is_abnormal": literal_eval,
        }

        datasets_as_dfs = {}
        datasets_as_dfs["train"] = pd.read_csv(os.path.join(self.path_full_dataset, "train.csv"), usecols=usecols,
                                               converters=converters)

        # val dataset has additional "reference_report" column
        usecols.append("reference_report")
        datasets_as_dfs["valid"] = pd.read_csv(os.path.join(self.path_full_dataset, "valid.csv"), usecols=usecols,
                                               converters=converters)

        total_num_samples_train = len(datasets_as_dfs["train"])
        total_num_samples_val = len(datasets_as_dfs["valid"])

        # compute new number of samples for both train and val
        new_num_samples_train = int(self.percentage_of_train_set_to_use * total_num_samples_train)
        new_num_samples_val = int(self.percentage_of_val_set_to_use * total_num_samples_val)

        self.log.info(f"Train: {new_num_samples_train} images")
        self.log.info(f"Val: {new_num_samples_val} images")

        with open(config_file_path, "a") as f:
            f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
            f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

        # limit the datasets to those new numbers
        datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
        datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

        raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
        raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

        return raw_train_dataset, raw_val_dataset

    def create_run_folder(self):
        """
        Run folder will contain:
            - a folder called "checkpoints" for the saved checkpoints
            - a folder called "tensorboard" for the saved tensorboard files
            - a folder called "generated_sentences_and_reports" that store the generated sentences and reports
            which were created at each evaluation
            - a txt file called "log_file", which stores information like OOMs that happened during training
            - a txt file called "run_config.txt", which stores the information specified in run_configurations.py
        """
        run_folder_path = os.path.join(self.path_runs_full_model, f"run_{self.run}")
        checkpoints_folder_path = os.path.join(run_folder_path, "checkpoints")
        tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
        generated_sentences_and_reports_folder_path = os.path.join(run_folder_path, "generated_sentences_and_reports")
        generated_sentences_folder_path = os.path.join(generated_sentences_and_reports_folder_path,
                                                       "generated_sentences")
        generated_reports_folder_path = os.path.join(generated_sentences_and_reports_folder_path, "generated_reports")
        log_file = os.path.join(run_folder_path, "log_file.txt")

        if os.path.exists(run_folder_path):
            if self.auto_delete_run_folder:
                self.log.info(f"Run_{self.run} Folder has already been automatically deleted at {run_folder_path}.")
                shutil.rmtree(run_folder_path)
            else:
                self.log.error(f"Folder to save run {self.run} already exists at {run_folder_path}.")
                self.log.error("Delete the folder or change the run number.")
                return None

        os.mkdir(run_folder_path)
        os.mkdir(checkpoints_folder_path)
        os.mkdir(tensorboard_folder_path)
        os.mkdir(generated_sentences_and_reports_folder_path)
        os.mkdir(generated_sentences_folder_path)
        os.mkdir(generated_reports_folder_path)

        self.log.info(f"Run {self.run} folder created at {run_folder_path}.")

        config_file_path = os.path.join(run_folder_path, "run_config.txt")
        config_parameters = {
            "COMMENT": f"Run Start Time: {self.run_start_time}",
            "SEED": self.seed,
            "PRETRAIN_WITHOUT_LM_MODEL": self.pretrain_without_lm_model,
            "IMAGE_INPUT_SIZE": self.image_input_size,
            "PERCENTAGE_OF_TRAIN_SET_TO_USE": self.percentage_of_train_set_to_use,
            "PERCENTAGE_OF_VAL_SET_TO_USE": self.percentage_of_val_set_to_use,
            "BATCH_SIZE": self.batch_size,
            "EFFECTIVE_BATCH_SIZE": self.effective_batch_size,
            "NUM_WORKERS": self.num_workers,
            "EPOCHS": self.epochs,
            "LR": self.lr,
            "EVALUATE_EVERY_K_BATCHES": self.evaluate_every_k_batches,
            "PATIENCE_LR_SCHEDULER": self.patience_lr_scheduler,
            "THRESHOLD_LR_SCHEDULER": self.threshold_lr_scheduler,
            "FACTOR_LR_SCHEDULER": self.factor_lr_scheduler,
            "COOLDOWN_LR_SCHEDULER": self.cooldown_lr_scheduler,
            "NUM_BEAMS": self.num_beams,
            "MAX_NUM_TOKENS_GENERATE": self.max_num_tokens_generate,
            "NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE": self.num_batches_of_generated_sentences_to_save_to_file,
            "NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE": self.num_batches_of_generated_reports_to_save_to_file,
            "NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION": self.num_batches_to_process_for_language_model_evaluation,
            "NUM_IMAGES_TO_PLOT": self.num_images_to_plot,
            "BERTSCORE_SIMILARITY_THRESHOLD": self.bertscore_similarity_threshold,
            "WEIGHT_OBJECT_DETECTOR_LOSS": self.weight_object_detector_loss,
            "WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS": self.weight_binary_classifier_region_selection_loss,
            "WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS": self.weight_binary_classifier_region_abnormal_loss,
            "WEIGHT_LANGUAGE_MODEL_LOSS": self.weight_language_model_loss,
        }

        with open(config_file_path, "w") as f:
            f.write(f"RUN {self.run}:\n")
            for param_name, param_value in config_parameters.items():
                f.write(f"\t{param_name}: {param_value}\n")

        with open(log_file,'w') as f:
            f.close()

        return checkpoints_folder_path, tensorboard_folder_path, config_file_path, generated_sentences_and_reports_folder_path, log_file

    def train(self):
        (
            checkpoints_folder_path, tensorboard_folder_path, config_file_path,
            generated_sentences_and_reports_folder_path,
            log_file) = self.create_run_folder()

        # the datasets still contain the untokenized phrases
        raw_train_dataset, raw_val_dataset = self.get_datasets(config_file_path)

        tokenizer = self.get_tokenizer()

        # tokenize the raw datasets
        tokenized_train_dataset, tokenized_val_dataset = self.get_tokenized_datasets(tokenizer, raw_train_dataset,
                                                                                     raw_val_dataset)

        train_transforms = self.get_transforms("train")
        val_transforms = self.get_transforms("val")

        train_dataset_complete = CustomDataset("train", tokenized_train_dataset, train_transforms, self.log)
        val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms, self.log)

        train_loader, val_loader = self.get_data_loaders(tokenizer, train_dataset_complete, val_dataset_complete)

        # resume_training = False
        checkpoint = None
        if self.path_to_best_classification_modules_weights != 'None':
            checkpoint = torch.load(
                self.path_to_best_classification_modules_weights,
                map_location=self.device
            )

        model = self.get_model(checkpoint)

        opt = AdamW(model.parameters(), lr=self.lr)
        scaler = torch.cuda.amp.GradScaler()

        current_epoch = 0
        overall_steps_taken = 0
        lowest_val_loss = np.inf

        # if resume_training:
        #     model.load_state_dict(checkpoint["model"])
        #     opt.load_state_dict(checkpoint["optimizer"])
        #     scaler.load_state_dict(checkpoint["scaler"])
        #     current_epoch = checkpoint["current_epoch"]
        #     overall_steps_taken = checkpoint["overall_steps_taken"]
        #     lowest_val_loss = checkpoint["lowest_val_loss"]

        lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=self.factor_lr_scheduler,
                                         patience=self.patience_lr_scheduler, threshold=self.threshold_lr_scheduler,
                                         cooldown=self.cooldown_lr_scheduler)
        writer = SummaryWriter(log_dir=tensorboard_folder_path)

        if checkpoint:
            del checkpoint

        self.log.info("Starting training!")

        self.train_model(
            model=model,
            train_dl=train_loader,
            val_dl=val_loader,
            optimizer=opt,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            current_epoch=current_epoch,
            epochs=self.epochs,
            overall_steps_taken=overall_steps_taken,
            lowest_val_loss=lowest_val_loss,
            checkpoints_folder_path=checkpoints_folder_path,
            tokenizer=tokenizer,
            generated_sentences_and_reports_folder_path=generated_sentences_and_reports_folder_path,
            writer=writer,
            log_file=log_file,
        )
