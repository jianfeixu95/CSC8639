import os
from collections import defaultdict
import albumentations as A
import cv2
import evaluate
import spacy
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from transformers import GPT2Tokenizer
from src.full_model.report_generation_model import ReportGenerationModel

class GeneratedReportsForImages(object):

    def __init__(
            self,
            mean,
            std,
            num_beams,
            image_input_size,
            path_to_best_full_model_weights,
            image_format,
            image_folder_paths,
            max_num_tokens_generate,
            bertscore_similarity_threshold,
            generated_reports_folder_path,
            path_to_best_object_detector_weights):
        self.mean = mean
        self.std = std
        self.bertscore_similarity_threshold = bertscore_similarity_threshold
        self.max_num_tokens_generate = max_num_tokens_generate
        self.num_beams = num_beams
        self.image_input_size = image_input_size
        self.path_to_best_full_model_weights = path_to_best_full_model_weights
        self.image_format = image_format
        self.image_folder_paths = image_folder_paths
        self.generated_reports_folder_path = generated_reports_folder_path
        self.path_to_best_object_detector_weights = path_to_best_object_detector_weights

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_tokenizer(self):
        checkpoint = "healx/gpt-2-pubmed-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def write_generated_reports_to_txt(self, images_paths, generated_reports, generated_reports_txt_path, type):
        if type == "single":
            with open(generated_reports_txt_path, "w") as f:
                f.write(f"Image path: {images_paths}\n")
                f.write(f"Generated report: {generated_reports}\n\n")
                f.write("=" * 30)
                f.write("\n\n")
        elif type == "multi":
            with open(generated_reports_txt_path, "w") as f:
                for image_path, report in zip(images_paths, generated_reports):
                    f.write(f"Image path: {image_path}\n")
                    f.write(f"Generated report: {report}\n\n")
                    f.write("=" * 30)
                    f.write("\n\n")

    def remove_duplicate_generated_sentences(self, generated_report, bert_score, sentence_tokenizer):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

        # use sentence tokenizer to separate the generated sentences
        gen_sents = sentence_tokenizer(generated_report).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        gen_sents = [sent.text for sent in gen_sents]

        # remove exact duplicates using a dict as an ordered set
        # note that dicts are insertion ordered as of Python 3.7
        gen_sents = list(dict.fromkeys(gen_sents))

        # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
        # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
        # to remove these "soft" duplicates, we use bertscore

        # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
        similar_generated_sents_to_be_removed = defaultdict(list)

        for i in range(len(gen_sents)):
            gen_sent_1 = gen_sents[i]

            for j in range(i + 1, len(gen_sents)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                bert_score_result = bert_score.compute(
                    lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                )

                if bert_score_result["f1"][0] > self.bertscore_similarity_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        generated_report = " ".join(
            sent
            for sent in gen_sents
            if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return generated_report

    def convert_generated_sentences_to_report(self, generated_sents_for_selected_regions, bert_score,
                                              sentence_tokenizer):
        generated_report = " ".join(sent for sent in generated_sents_for_selected_regions)

        generated_report = self.remove_duplicate_generated_sentences(generated_report, bert_score, sentence_tokenizer)
        return generated_report

    def get_report_for_image(self, model, image_tensor, tokenizer, bert_score, sentence_tokenizer):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.generate(
                image_tensor.to(self.device, non_blocking=True),
                max_length=self.max_num_tokens_generate,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        beam_search_output, _, _, _ = output

        generated_sents_for_selected_regions = tokenizer.batch_decode(
            beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )  # list[str]

        generated_report = self.convert_generated_sentences_to_report(
            generated_sents_for_selected_regions, bert_score, sentence_tokenizer
        )  # str

        return generated_report

    def get_image_tensor(self, image_path):
        # cv2.imread by default loads an image with 3 channels
        # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape (3056, 2544)

        val_test_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.image_input_size, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=self.image_input_size, min_width=self.image_input_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

        transform = val_test_transforms(image=image)
        image_transformed = transform["image"]  # shape (1, 512, 512)
        image_transformed_batch = image_transformed.unsqueeze(0)  # shape (1, 1, 512, 512)

        return image_transformed_batch

    def get_model(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
        )

        # if there is a key error when loading checkpoint, try uncommenting down below
        # since depending on the torch version, the state dicts may be different
        # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
        # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")
        model = ReportGenerationModel(self.path_to_best_object_detector_weights,pretrain_without_lm_model=True)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device, non_blocking=True)
        model.eval()

        del checkpoint

        return model


    def get_jpg_file_paths(self):
        jpg_file_paths = []
        for root, dirs, files in os.walk(self.image_folder_paths):
            for file in files:
                if file.endswith(self.image_format):
                    jpg_file_paths.append(os.path.join(root, file))
        return jpg_file_paths

    def generate(self):
        model = self.get_model(self.path_to_best_full_model_weights)
        print("Model instantiated.")

        bert_score = evaluate.load("bertscore")
        sentence_tokenizer = spacy.load("en_core_web_trf")
        tokenizer = self.get_tokenizer()

        generated_reports = []
        images_paths = self.get_jpg_file_paths()

        for image_path in tqdm(images_paths):
            generated_reports_txt_path = os.path.join(
                self.generated_reports_folder_path,
                f"report_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
            )
            image_tensor = self.get_image_tensor(image_path)
            generated_report = self.get_report_for_image(model, image_tensor, tokenizer, bert_score, sentence_tokenizer)
            self.write_generated_reports_to_txt(image_path, generated_report, generated_reports_txt_path,"single")
            generated_reports.append(generated_report)

        self.write_generated_reports_to_txt(
            images_paths, generated_reports,
            os.path.join(
                self.generated_reports_folder_path,
                'reports.txt'
            ), "multi")
