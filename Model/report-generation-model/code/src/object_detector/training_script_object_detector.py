import shutil
from ast import literal_eval
from copy import deepcopy
import logging
import os
import random
from typing import List, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset.constants import ANATOMICAL_REGIONS
from src.object_detector.custom_image_dataset_object_detector import CustomImageDataset
from src.object_detector.object_detector import ObjectDetector
from datetime import datetime

class TrainObjectDetector(object):
    def __init__(
            self,
            config: dict):
        self.path_full_dataset = config['dataset']['path_full_dataset']
        self.path_runs_object_detector = config['runs']['path_runs_object_detector']
        self.run = config['train']['object_detector']['run']
        self.seed = config['train']['object_detector']['seed']
        self.epochs = config['train']['object_detector']['epochs']
        self.num_workers = config['train']['object_detector']['num_workers']
        self.batch_size = config['train']['object_detector']['batch_size']
        self.effective_batch_size = config['train']['object_detector']['effective_batch_size']
        self.evaluate_every_k_steps = config['train']['object_detector']['evaluate_every_k_steps']
        self.lr = config['train']['object_detector']['lr']
        self.patience_lr_scheduler = config['train']['object_detector']['patience_lr_scheduler']
        self.threshold_lr_scheduler = config['train']['object_detector']['threshold_lr_scheduler']
        self.factor_lr_scheduler = config['train']['object_detector']['factor_lr_scheduler']
        self.cooldown_lr_scheduler = config['train']['object_detector']['cooldown_lr_scheduler']
        self.image_input_size = config['train']['object_detector']['image_input_size']
        self.percentage_of_train_set_to_use = config['train']['object_detector']['percentage_of_train_set_to_use']
        self.percentage_of_val_set_to_use = config['train']['object_detector']['percentage_of_val_set_to_use']
        self.auto_delete_run_folder = config['runs']['auto_delete_run_folder']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
        self.log = logging.getLogger(__name__)
        self.run_start_time = datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p")
        self.set_seed()

    def set_seed(self):
        # set the seed value for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def get_title(self, region_set, region_indices, region_colors, class_detected_img):
        # region_set always contains 6 region names (except for region_set_5)

        # get a list of 6 boolean values that specify if that region was detected
        class_detected = [class_detected_img[region_index] for region_index in region_indices]

        # add color_code to region name (e.g. "(r)" for red)
        # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
        region_set = [region + f" ({color})" if cls_detect else region + f" ({color}, nd)" for region, color, cls_detect in zip(region_set, region_colors, class_detected)]

        # add a line break to the title, as to not make it too long
        return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


    def plot_box(self, box, ax, clr, linestyle, class_detected=True):
        x0, y0, x1, y1 = box
        h = y1 - y0
        w = x1 - x0
        ax.add_artist(
            plt.Rectangle(
                xy=(x0, y0),
                height=h,
                width=w,
                fill=False,
                color=clr,
                linewidth=1,
                linestyle=linestyle
            )
        )

        # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding class was not detected)
        if not class_detected:
            ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


    def plot_gt_and_pred_bboxes_to_tensorboard(self, writer, overall_steps_taken, images, detections, targets, class_detected, num_images_to_plot=2):
        # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes_batch = detections["top_region_boxes"]

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
        # gt_boxes is of shape [batch_size x 29 x 4]
        gt_boxes_batch = torch.stack([t["boxes"] for t in targets], dim=0)

        # plot 6 regions at a time, as to not overload the image with boxes (except for region_set_5, which has 5 regions)
        # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
        region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
        region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
        region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
        region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
        region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

        regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]

        for num_img in range(num_images_to_plot):
            image = images[num_img].cpu().numpy().transpose(1, 2, 0)

            gt_boxes_img = gt_boxes_batch[num_img]
            pred_boxes_img = pred_boxes_batch[num_img]
            class_detected_img = class_detected[num_img].tolist()

            for num_region_set, region_set in enumerate(regions_sets):
                fig = plt.figure(figsize=(8, 8))
                ax = plt.gca()

                plt.imshow(image, cmap='gray')
                plt.axis('off')

                region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
                region_colors = ["b", "g", "r", "c", "m", "y"]

                if num_region_set == 4:
                    region_colors.pop()

                for region_index, color in zip(region_indices, region_colors):
                    box_gt = gt_boxes_img[region_index].tolist()
                    box_pred = pred_boxes_img[region_index].tolist()
                    box_class_detected = class_detected_img[region_index]

                    self.plot_box(box_gt, ax, clr=color, linestyle="solid", class_detected=box_class_detected)

                    # only plot predicted box if class was actually detected
                    if box_class_detected:
                        self.plot_box(box_pred, ax, clr=color, linestyle="dashed")

                title = self.get_title(region_set, region_indices, region_colors, class_detected_img)
                ax.set_title(title)

                writer.add_figure(f"img_{num_img}_region_set_{num_region_set}", fig, overall_steps_taken)


    def compute_box_area(self, box):
        """
        Calculate the area of a box given the 4 corner values.

        Args:
            box (Tensor[batch_size x 29 x 4])

        Returns:
            area (Tensor[batch_size x 29])
        """
        x0 = box[..., 0]
        y0 = box[..., 1]
        x1 = box[..., 2]
        y1 = box[..., 3]

        return (x1 - x0) * (y1 - y0)


    def compute_intersection_and_union_area_per_class(self, detections, targets, class_detected):
        # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes = detections["top_region_boxes"]

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
        # gt_boxes is of shape [batch_size x 29 x 4]
        gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

        # below tensors are of shape [batch_size x 29]
        x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
        y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
        x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
        y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

        # intersection_boxes is of shape [batch_size x 29 x 4]
        intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

        # below tensors are of shape [batch_size x 29]
        intersection_area = self.compute_box_area(intersection_boxes)
        pred_area = self.compute_box_area(pred_boxes)
        gt_area = self.compute_box_area(gt_boxes)

        # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
        valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

        # also there is no intersection if the class was not detected by object detector
        valid_intersection = torch.logical_and(valid_intersection, class_detected)

        # set all non-valid intersection areas to 0
        intersection_area = torch.where(valid_intersection, intersection_area, torch.tensor(0, dtype=intersection_area.dtype, device=intersection_area.device))

        union_area = (pred_area + gt_area) - intersection_area

        # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
        intersection_area = torch.sum(intersection_area, dim=0)
        union_area = torch.sum(union_area, dim=0)

        return intersection_area, union_area


    def get_val_loss_and_other_metrics(self, model, val_dl, writer, overall_steps_taken):
        """
        Args:
            model (nn.Module): The input model to be evaluated.
            val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.
            writer (tensorboardX.SummaryWriter.writer): Writer used to plot gt and predicted bboxes of first couple of image in val set
            overall_steps_taken: for tensorboard

        Returns:
            val_loss (float): val loss for val set
            avg_num_detected_classes_per_image (float): since it's possible that certain classes/regions of all 29 regions are not detected in an image,
            this metric counts how many classes are detected on average for an image. Ideally, this number should be 29.0
            avg_detections_per_class (list[float]): this metric counts how many times a class was detected in an image on average. E.g. if the value is 1.0,
            then the class was detected in all images of the val set
            avg_iou_per_class (list[float]): average IoU per class computed over all images in val set
        """
        # PyTorch implementation only return losses in train mode, and only detections in eval mode
        # see https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn/65347721#65347721
        # my model is modified to return losses, detections and class_detected in eval mode
        # see forward method of object detector class for more information
        model.eval()

        val_loss = 0.0

        num_images = 0

        # tensor for accumulating the number of times a class is detected over all images (will be divided by num_images at the end of get average)
        sum_class_detected = torch.zeros(29, device=self.device)

        # tensor for accumulating the intersection area of each class (will be divided by union area of each class at the end of get the IoU for each class)
        sum_intersection_area_per_class = torch.zeros(29, device=self.device)

        # tensor for accumulating the union area of each class (will divide the intersection area of each class at the end of get the IoU for each class)
        sum_union_area_per_class = torch.zeros(29, device=self.device)

        with torch.no_grad():
            for batch_num, batch in tqdm(enumerate(val_dl)):
                # "targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
                # "boxes" maps to a tensor of shape [29 x 4] and "labels" maps to a tensor of shape [29]
                # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 29] (starting at 1, since 0 is background)
                images, targets = batch.values()

                batch_size = images.size(0)
                num_images += batch_size

                images = images.to(self.device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                # detections is a dict with keys "top_region_boxes" and "top_scores"
                # "top_region_boxes" maps to a tensor of shape [batch_size x 29 x 4]
                # "top_scores" maps to a tensor of shape [batch_size x 29]

                # class_detected is a tensor of shape [batch_size x 29]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss_dict, detections, class_detected = model(images, targets)

                # sum up all 4 losses
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item() * batch_size

                # sum up detections for each class
                sum_class_detected += torch.sum(class_detected, dim=0)

                # compute intersection and union area for each class and add them to the sum
                intersection_area_per_class, union_area_per_class = self.compute_intersection_and_union_area_per_class(detections, targets, class_detected)
                sum_intersection_area_per_class += intersection_area_per_class
                sum_union_area_per_class += union_area_per_class

                if batch_num == 0:
                    self.plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detections, targets, class_detected, num_images_to_plot=2)

        val_loss /= len(val_dl)
        avg_num_detected_classes_per_image = torch.sum(sum_class_detected / num_images).item()
        avg_detections_per_class = (sum_class_detected / num_images).tolist()
        avg_iou_per_class = (sum_intersection_area_per_class / sum_union_area_per_class).tolist()

        return val_loss, avg_num_detected_classes_per_image, avg_detections_per_class, avg_iou_per_class


    def log_stats_to_console(
            self,
            train_loss,
            val_loss,
            epoch,
    ):
        self.log.info(f"Epoch: {epoch}:")
        self.log.info(f"\tTrain loss: {train_loss:.3f}")
        self.log.info(f"\tVal loss: {val_loss:.3f}")


    def train_model(
            self,
            model,
            train_dl,
            val_dl,
            optimizer,
            scaler,
            lr_scheduler,
            epochs,
            weights_folder_path,
            writer
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
        patience: int
            Number of epochs to wait for val loss to decrease.
            If patience is exceeded, then training is stopped early.
        weights_folder_path: str
            Path to folder where best weights will be saved.
        writer: torch.utils.tensorboard.SummaryWriter
            Writer for logging values to tensorboard.

        Returns
        -------
        None, but saves model with the overall lowest val loss at the end of every epoch.
        """
        lowest_val_loss = np.inf

        # the best_model_state is the one where the val loss is the lowest overall
        best_model_state = None

        overall_steps_taken = 0  # for logging to tensorboard

        # for gradient accumulation
        ACCUMULATION_STEPS = self.effective_batch_size // self.batch_size

        for epoch in range(epochs):
            self.log.info(f"Training epoch {epoch}!")

            train_loss = 0.0
            steps_taken = 0
            for num_batch, batch in tqdm(enumerate(train_dl)):
                # batch is a dict with keys "images" and "targets"
                images, targets = batch.values()

                batch_size = images.size(0)

                images = images.to(self.device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images, targets)

                    # sum up all 4 losses
                    loss = sum(loss for loss in loss_dict.values())

                scaler.scale(loss).backward()

                if (num_batch + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * batch_size
                steps_taken += 1
                overall_steps_taken += 1

                # evaluate every k steps and also at the end of an epoch
                if steps_taken >= self.evaluate_every_k_steps or (num_batch + 1) == len(train_dl):
                    self.log.info(f"Evaluating at step {overall_steps_taken}!")

                    # normalize the train loss by steps_taken
                    train_loss /= steps_taken

                    val_loss, avg_num_detected_classes_per_image, avg_detections_per_class, avg_iou_per_class = self.get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken)

                    writer.add_scalars("_loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)
                    writer.add_scalar("avg_num_predicted_classes_per_image", avg_num_detected_classes_per_image, overall_steps_taken)

                    # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
                    anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]

                    for class_, avg_detections_class in zip(anatomical_regions, avg_detections_per_class):
                        writer.add_scalar(f"num_preds_{class_}", avg_detections_class, overall_steps_taken)

                    for class_, avg_iou_class in zip(anatomical_regions, avg_iou_per_class):
                        writer.add_scalar(f"iou_{class_}", avg_iou_class, overall_steps_taken)

                    current_lr = float(optimizer.param_groups[0]["lr"])
                    writer.add_scalar("lr", current_lr, overall_steps_taken)

                    self.log.info(f"Metrics evaluated at step {overall_steps_taken}!")

                    # set the model back to training
                    model.train()

                    # decrease lr if val loss has not decreased after certain number of evaluations
                    lr_scheduler.step(val_loss)

                    if val_loss < lowest_val_loss:
                        lowest_val_loss = val_loss
                        best_epoch = epoch
                        best_model_save_path = os.path.join(
                            weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}.pth"
                        )
                        best_model_state = deepcopy(model.state_dict())

                    # log to console at the end of an epoch
                    if (num_batch + 1) == len(train_dl):
                        self.log_stats_to_console(train_loss, val_loss, epoch)

                    # reset values
                    train_loss = 0.0
                    steps_taken = 0

            # save the current best model weights at the end of each epoch
            torch.save(best_model_state, best_model_save_path)

        self.log.info("Finished training!")
        self.log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
        return None


    def collate_fn(self, batch: List[Dict[str, Tensor]]):
        # each dict in batch (which is a list) is for a single image and has the keys "image", "boxes", "labels"

        # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
        # otherwise, whole training loop will stop (even if only 1 image fails to open)
        batch = list(filter(lambda x: x is not None, batch))

        image_shape = batch[0]["image"].size()
        # allocate an empty images_batch tensor that will store all images of the batch
        images_batch = torch.empty(size=(len(batch), *image_shape))

        for i, sample in enumerate(batch):
            # remove image tensors from batch and store them in dedicated images_batch tensor
            images_batch[i] = sample.pop("image")

        # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
        targets = batch

        # create a new batch variable to store images_batch and targets
        batch_new = {}
        batch_new["images"] = images_batch
        batch_new["targets"] = targets

        return batch_new

    def seed_worker(self,worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data_loaders(self,train_dataset, val_dataset):
        g = torch.Generator()
        g.manual_seed(self.seed)

        train_loader = DataLoader(train_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn=self.seed_worker, generator=g, pin_memory=True)
        val_loader = DataLoader(val_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader


    def get_transforms(self,dataset: str):
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
                A.PadIfNeeded(min_height=self.image_input_size, min_width=self.image_input_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
        )

        # don't apply data augmentations to val and test set
        val_test_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.image_input_size, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=self.image_input_size, min_width=self.image_input_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
        )

        if dataset == "train":
            return train_transforms
        else:
            return val_test_transforms


    def get_datasets_as_dfs(self, config_file_path):
        usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]

        # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
        # the literal_eval func to convert them to python lists
        converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

        datasets_as_dfs = {dataset: os.path.join(self.path_full_dataset, dataset) + ".csv" for dataset in ["train", "valid"]}
        datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

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

        return datasets_as_dfs


    def create_run_folder(self):
        """
        Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
        as well as a config file that specifies the overall parameters used for training.
        """
        run_folder_path = os.path.join(self.path_runs_object_detector, f"run_{self.run}")
        weights_folder_path = os.path.join(run_folder_path, "weights")
        tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")

        if os.path.exists(run_folder_path):
            if self.auto_delete_run_folder:
                self.log.info(f"Run_{self.run} folder has already been automatically deleted at {run_folder_path}.")
                shutil.rmtree(run_folder_path)
            else:
                self.log.error(f"Folder to save run {self.run} already exists at {run_folder_path}.")
                self.log.error("Delete the folder or change the run number.")
                return None

        os.mkdir(run_folder_path)
        os.mkdir(weights_folder_path)
        os.mkdir(tensorboard_folder_path)

        self.log.info(f"Run {self.run} folder created at {run_folder_path}.")

        config_file_path = os.path.join(run_folder_path, "run_config.txt")
        config_parameters = {
            "COMMENT": f"Run Start Time: {self.run_start_time}",
            "SEED": self.seed,
            "IMAGE_INPUT_SIZE": self.image_input_size,
            "PERCENTAGE_OF_TRAIN_SET_TO_USE": self.percentage_of_train_set_to_use,
            "PERCENTAGE_OF_VAL_SET_TO_USE": self.percentage_of_val_set_to_use,
            "BATCH_SIZE": self.batch_size,
            "EFFECTIVE_BATCH_SIZE": self.effective_batch_size,
            "NUM_WORKERS": self.num_workers,
            "EPOCHS": self.epochs,
            "LR": self.lr,
            "EVALUATE_EVERY_K_STEPS": self.evaluate_every_k_steps,
            "PATIENCE_LR_SCHEDULER": self.patience_lr_scheduler,
            "THRESHOLD_LR_SCHEDULER": self.threshold_lr_scheduler,
            "FACTOR_LR_SCHEDULER": self.factor_lr_scheduler,
            "COOLDOWN_LR_SCHEDULER": self.cooldown_lr_scheduler,
        }

        with open(config_file_path, "w") as f:
            f.write(f"RUN {self.run}:\n")
            for param_name, param_value in config_parameters.items():
                f.write(f"\t{param_name}: {param_value}\n")

        return weights_folder_path, tensorboard_folder_path, config_file_path

    def train(self):
        weights_folder_path, tensorboard_folder_path, config_file_path = self.create_run_folder()
        datasets_as_dfs = self.get_datasets_as_dfs(config_file_path)
        train_transforms = self.get_transforms("train")
        val_transforms = self.get_transforms("val")
        train_dataset = CustomImageDataset(datasets_as_dfs["train"], train_transforms)
        val_dataset = CustomImageDataset(datasets_as_dfs["valid"], val_transforms)
        train_loader, val_loader = self.get_data_loaders(train_dataset, val_dataset)
        model = ObjectDetector(return_feature_vectors=False)
        model.to(self.device, non_blocking=True)
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        opt = AdamW(model.parameters(), lr=eval(self.lr))
        lr_scheduler = ReduceLROnPlateau(
            opt, mode="min",
            factor=self.factor_lr_scheduler, patience=self.patience_lr_scheduler,
            threshold= eval(self.threshold_lr_scheduler), cooldown=self.cooldown_lr_scheduler)
        writer = SummaryWriter(log_dir=tensorboard_folder_path)
        self.log.info("\nStarting training!\n")
        self.train_model(
            model=model,
            train_dl=train_loader,
            val_dl=val_loader,
            optimizer=opt,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            epochs=self.epochs,
            weights_folder_path=weights_folder_path,
            writer=writer
        )
