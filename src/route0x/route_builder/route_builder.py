from __future__ import annotations
import sys
import os
import json
import pkg_resources
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from collections import Counter
from datetime import datetime
import joblib
import subprocess
import platform
import time
import csv
import random


import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from sklearn.metrics import classification_report
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sentence_transformers.losses import CoSENTLoss, AnglELoss
from tqdm.auto import tqdm

import requests
from ollama import Client
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16

import chardet
import ollama
import torch
from setfit import SetFitModel

import nlpaug.augmenter.char as nac


from .outlier_detector import OutlierDetector
from .vector_db import VectorDB
from .unified_llm_caller import UnifiedLLM
from .losses import PairwiseArcFaceFocalLoss, ScaledAnglELoss, BinaryLabelTripletMarginLoss

import warnings
warnings.filterwarnings("ignore")

LOSS_FUNCTIONS = {
            "PairwiseArcFaceFocalLoss": PairwiseArcFaceFocalLoss,
            "ScaledAngleLoss": ScaledAnglELoss,
            "BinaryLabelTripletMarginLoss": BinaryLabelTripletMarginLoss,
            "CoSENTLoss": CoSENTLoss,
            "AnglELoss": AnglELoss
        }


class Route:
    def __init__(self, route_name, samples):
        """
        Represents a single route with its respective samples.
        """
        self.route_name = route_name
        self.samples = samples

class RouteBuilderRequest:
    def __init__(self, routes):
        """
        Represents a build request that contains multiple routes and their respective samples.
        """
        self.routes = routes

    def to_csv(self, output_file):
        """
        Converts the request data into a CSV format with 'text' and 'label' columns.
        """
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["text", "label"])
            for route in self.routes:
                for sample in route.samples:
                    writer.writerow([sample, route.route_name])

class RouteBuilder:
    def __init__(self,
                 build_request: RouteBuilderRequest = None,
                 hf_dataset: Optional[str] = None,
                 label_column: str = None,
                 text_column: str = None,
                 train_path: Optional[str] = None,
                 eval_path: Optional[str] = None,
                 model_name: str = None,
                 epochs: int = None,
                 max_steps: int = None,
                 batch_size: int = None,
                 lr: float = None,
                 loss_funct_name = None,
                 warmup_proportion: float = None,
                 min_samples: int = None,
                 samples_per_route: Optional[int] = None,
                 routes: Optional[List[str]] = None,
                 route_template: str = None,
                 domain: Optional[str] = None,
                 llm_name: str = None,
                 instruct_llm: str = None,
                 enable_id_oos_gen: bool = None,
                 enable_synth_data_gen: bool = None,
                 max_query_len: int = None,
                 seed: int = None,
                 device: str = None,
                 oos_label: str = None,
                 expected_oos_proportion: float = None,
                 nn_for_oos_detection: int = None,
                 skip_eval: bool = False,
                 add_additional_invalid_routes: bool = False,
                 invalid_routes: Optional[List[str]] = None,
                 only_oos_head: bool = False,
                 log_level: str = "info",
                 do_quantise: bool = False,
                 add_typo_robustness: bool = False):
                 
        """
        Initializes the RouteBuilder with configuration parameters.
        """
        config_path = pkg_resources.resource_filename('route0x', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.train_path = train_path if train_path is not None else config.get("train_path")
        self.eval_path = eval_path if eval_path is not None else config.get("eval_path")
        self.model_name = model_name if model_name is not None else config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        self.epochs = epochs if epochs is not None else config.get("epochs")
        self.max_steps = max_steps if max_steps is not None else config.get("max_steps")
        self.batch_size = batch_size if batch_size is not None else config.get("batch_size")
        self.output_dir = config.get("output_dir", "output")
        self.seed = seed if seed is not None else config.get("seed", "1234")
        self.lr = lr if lr is not None else config.get("lr")
        self.loss_funct_name = loss_funct_name if loss_funct_name is not None else config.get("loss_funct_name")
        self.warmup_proportion = warmup_proportion if warmup_proportion is not None else config.get("warmup_proportion")
        self.min_samples = min_samples if min_samples is not None else config.get("min_samples")
        self.samples_per_route = samples_per_route if samples_per_route is not None else config.get("samples_per_route")
        self.routes = routes if routes is not None else config.get("routes")
        self.route_template = route_template if route_template is not None else config.get("route_template")
        self.domain = domain if domain is not None else config.get("domain")
        self.llm_name = llm_name if llm_name is not None else config.get("llm_name")
        self.instruct_llm = instruct_llm if instruct_llm is not None else config.get("instruct_llm")
        self.enable_id_oos_gen = enable_id_oos_gen if enable_id_oos_gen is not None else config.get("enable_id_oos_gen")
        self.enable_synth_data_gen = enable_synth_data_gen if enable_synth_data_gen is not None else config.get("enable_synth_data_gen")
        self.max_query_len = max_query_len if max_query_len is not None else config.get("max_query_len")
        self.hf_dataset = hf_dataset if hf_dataset is not None else config.get("hf_dataset")
        self.label_column = label_column if label_column is not None else config.get("label_column")
        self.text_column = text_column if text_column is not None else config.get("text_column")
        self.oos_label = oos_label if oos_label is not None else config.get("oos_label")
        self.add_additional_invalid_routes = add_additional_invalid_routes if add_additional_invalid_routes else config.get("add_additional_invalid_routes")
        self.invalid_routes = invalid_routes if invalid_routes else config.get("invalid_routes")
        self.expected_oos_proportion = expected_oos_proportion if expected_oos_proportion else config.get("expected_oos_proportion")
        self.nn_for_oos_detection = nn_for_oos_detection if nn_for_oos_detection else config.get("nn_for_oos_detection")
        self.skip_eval = skip_eval
        self.only_oos_head = only_oos_head
        self.add_typo_robustness = add_typo_robustness if add_typo_robustness else config.get("add_typo_robustness")
        self.build_request = build_request if build_request else None
        self.fallback_samples_per_route = config.get("fallback_samples_per_route")
        self.device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu") if not device else device
        self.do_quantise = do_quantise

        self.local_llm_host = config.get("local_llm_host")
        self.client = Client(host=self.local_llm_host)
        self.HOSTED_LLM_FAMILIES = config.get("hosted_llm_families")
        self.LOGO = """
            ____             _        _____           
            |  _ \ ___  _   _| |_ ___|  _  | __  __
            | |_) / _ \| | | | __/ _ \ | | |\ \/ /
            |  _ < (_) | |_| | ||  __/ |_| | >  < 
            |_| \_\___/ \__,_|\__\___|_____| /_/\_\\

                                                           
        Low Latency, High Accuracy, Custom Query routers for Humans and Agents.
        """

        self.logger = self._setup_logger(log_level)
        self.logger.info(f"Using device -  {self.device}")
    
    def show_defaults(self):
        return {k: v for k, v in vars(self).items() if self._is_json_serializable(v)}  
    
    def _setup_logger(self, level_str) -> logging.Logger:
        """
        Sets up the logger for the RouteBuilder.
        """
        
        level_mapping = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warn': logging.WARN,
            'error': logging.ERROR,
        }

        level = level_mapping.get(level_str.lower(), logging.INFO) 
        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger

    def _detect_encoding(self, file_path: str) -> str:
        """
        Detects the encoding of a file.
        """
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']

    def _load_data(self, 
                  file_path: Optional[str] = None, 
                  hf_dataset: Optional[str] = None, 
                  label_column: str = "label", 
                  text_column: str = "text", 
                  train_samples: int = 12, 
                  label_mapping = None,
                  ) -> Optional[Dataset]:
        """
        Loads data from a file or Hugging Face dataset.
        """
        if hf_dataset:
            train_path, eval_path = self._load_hf_dataset(hf_dataset, label_column, text_column, train_samples, label_mapping, self.route_template)
            return self._load_data(train_path), self._load_data(eval_path)
        
        if file_path is None:
            return None
        path = Path(file_path)
        try:
            if path.suffix == '.csv':
                encoding = self._detect_encoding(file_path)
                df = pd.read_csv(path, delimiter=",", encoding=encoding)
                df[label_column] = df[label_column].apply(lambda x: x.title() if x != self.oos_label else x)
                if self.route_template is not None:
                    df[label_column] = df[label_column].apply(lambda x: self.route_template.format(x) if x != self.oos_label else x )
            elif path.suffix == '.jsonl':
                with open(path, 'r', encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format. Please use CSV or JSONL.")
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Data must contain 'text' and 'label' columns.")
            return Dataset.from_pandas(df)
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_hf_dataset(self, dataset_name: str, 
                        label_column: str = "label", 
                        text_column: str = "text", 
                        train_samples: int = 8, 
                        label_mapping=None,
                        route_template=None
                        ):
        """
        Loads a dataset from Hugging Face, samples it, and saves as CSV files with label mapping.
        """
        if os.path.isdir(dataset_name):
            dataset = load_from_disk(dataset_name)
        else:
            dataset = load_dataset(dataset_name)
        dataset_with_indices = dataset["train"].add_column("index", list(range(len(dataset["train"]))))
        train_dataset = sample_dataset(dataset_with_indices, label_column=label_column, num_samples=train_samples, seed=self.seed)
        eval_dataset = None
        if "test" in dataset:    
            eval_dataset = sample_dataset(dataset["test"], label_column=label_column, num_samples=9999, seed=self.seed)
        if "validation" in dataset:
            eval_dataset = concatenate_datasets([eval_dataset, sample_dataset(dataset["validation"], label_column=label_column, num_samples=9999) ])
        train_indices = train_dataset["index"]
        eval_dataset = concatenate_datasets([eval_dataset,  dataset_with_indices.filter(lambda example: example["index"] not in train_indices)])
        eval_dataset = eval_dataset.remove_columns(["index"])  
        train_dataset.remove_columns(["index"])
        if label_mapping:
            train_df = pd.DataFrame({
                "text": train_dataset[text_column],
                "label": [route_template.format(label_mapping[label].replace("_", " ")) for label in train_dataset[label_column]]  
            })
            eval_df = pd.DataFrame({
                "text": eval_dataset[text_column],
                "label": [route_template.format(label_mapping[label].replace("_", " "))for label in eval_dataset[label_column]]  
            })
        else:
            train_df = pd.DataFrame({
                "text": train_dataset[text_column],
                "label": [route_template.format(label.replace("_", " ")) for label in train_dataset[label_column]]
            })
            eval_df = pd.DataFrame({
                "text": eval_dataset[text_column],
                "label": [route_template.format(label.replace("_", " ")) for label in eval_dataset[label_column]]
            })
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{dataset_name}_{timestamp}".replace("/", "_").replace("-", "_")    
        train_path = f"{prefix}_train.csv"
        eval_path = f"{prefix}_eval.csv"
        train_df.to_csv(train_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        return train_path, eval_path

    def _assess_data(self, dataset: Dataset, min_samples_per_class: int = 12) -> bool:
        """
        Assesses if the dataset has sufficient samples per class.
        """
        labels = dataset['label']
        label_counts = Counter(labels)
        insufficient_labels = {label: count for label, count in label_counts.items() if count < min_samples_per_class}
        if insufficient_labels:
            self.logger.warn("Warning: The following labels have insufficient samples:")
            for label, count in insufficient_labels.items():
                self.logger.info(f"  {label}: {count}/{min_samples_per_class}")
            return False
        return True

    def _split_synthetic_data(self, synthetic_df, min_samples_per_label):
        """
        Splits synthetic data into training and evaluation sets based on a minimum number of samples per label.
        Ensures that user samples are always part of the training set.
        """
        user_samples = synthetic_df[synthetic_df['is_user_sample'] == True] 
        synthetic_samples = synthetic_df[synthetic_df['is_user_sample'] == False] 
        self.logger.debug(user_samples.groupby('label').size())
        self.logger.debug(synthetic_samples.groupby('label').size())

        train_data = []
        eval_data = []

        for label in synthetic_df['label'].unique():
            # Get user and synthetic samples for the current label
            user_samples_for_label = user_samples[user_samples['label'] == label]
            synthetic_samples_for_label = synthetic_samples[synthetic_samples['label'] == label]
            self.logger.debug(f"synthetic_samples_for_label, {synthetic_samples_for_label}")

            # Calculate how many more samples we need from synthetic to reach min_samples_per_label
            user_sample_count = len(user_samples_for_label)
            remaining_samples_needed = max(0, min_samples_per_label - user_sample_count)
            self.logger.debug(f"for {label} needed is {remaining_samples_needed}")

            if remaining_samples_needed > 0:
                try:
                    train_size = min(remaining_samples_needed, len(synthetic_samples_for_label))
                    synthetic_train_samples, synthetic_eval_samples = train_test_split(
                            synthetic_samples_for_label,
                            train_size=train_size,
                            stratify=synthetic_samples_for_label['label'],
                            random_state=self.seed
                        )
                except Exception as e:
                    self.logger.error(e)    
            else:
                synthetic_train_samples = pd.DataFrame()  
                synthetic_eval_samples = synthetic_samples_for_label

            train_data.append(user_samples_for_label)
            train_data.append(synthetic_train_samples)
            eval_data.append(synthetic_eval_samples)

        # Concatenate the train and eval data from all labels
        train_data = pd.concat(train_data)
        eval_data = pd.concat(eval_data)

        return train_data, eval_data

    def _get_llm_provider(self, input_string):
        """
        Returns the provider based on the family of models in the input string.
        """
        input_string_lower = input_string.lower()
        for provider, families in self.HOSTED_LLM_FAMILIES.items():
            if any(family in input_string_lower for family in families):
                return provider
        return None 

    def _generate_synthetic_queries(
                                    self,
                                    labels: List[str],
                                    domain: str,
                                    samples_per_route: int,
                                    route_template: str,
                                    model: str = "llama3.1",
                                    prefix: str = None,
                                    user_instructions: str = "",
                                    train_samples: Optional[Dataset] = None,
                                    make_eval_samples: bool = True
                                ) -> Tuple[str, str]:
        """
        Generates synthetic queries using an LLM in N stages
        """
        synthetic_data = []
        train_file = os.path.join(self.synthetic_data_dir, f"{prefix}_train.csv")
        eval_file = os.path.join(self.synthetic_data_dir,f"{prefix}_eval.csv")
        ZERO_SHOT_SYSTEM_PROMPT = """You are an expert synthetic query generator for given labels. For a request like below, Request: Generate minimum of 2 short, high-quality, realistic queries with domain terminology for the label: 'Product Support' in 'e-commerce' domain without using the words in the label or domain the queries, Respond with a JSON list like below: ["What's your policy on software updates for electronic products?", "Do you provide warranty and spare parts for your products", "When can I expect my refund?"]. A Syntactically correct, well-formed JSON List as response is enough. No explanations or foot notes of any sort strictly NOT needed."""
        FEW_SHOT_SYSTEM_PROMPT = """You are an expert synthetic query generator for given labels. You will be provided with example queries for a specific label. Your task is to generate additional queries faithful to the domain and label, expanding on the given examples. Always respond with a JSON list of new queries, like: ["New query 1", "New query 2", "New query 3"]. Do not repeat the example queries. A syntactically correct, well-formed JSON List as response is enough. No explanations or footnotes of any sort are needed."""
        sample_dict = {}
        
        if train_samples is not None and len(train_samples) > 0:
            for sample in train_samples:
                if sample['label'] not in sample_dict:
                    sample_dict[sample['label']] = []
                sample_dict[sample['label']].append(sample['text'])
            system_prompt = FEW_SHOT_SYSTEM_PROMPT
        else:
            system_prompt = ZERO_SHOT_SYSTEM_PROMPT

        llm = UnifiedLLM(self.llm_provider, model, system_prompt=system_prompt)
        
        labels = sample_dict.keys() if len(sample_dict) > 0 else labels

        generate_stages = lambda N: [10] + [20 for _ in range(N - 1)]
        generate_phrases = lambda N: [""] + ["lexically diverse," for _ in range(N - 1)]
        gen_stage_count = (samples_per_route - 10) // 20 if samples_per_route > 10 else 0
        stage_wise_counts = generate_stages(gen_stage_count + 1)
        stage_wise_phrases = generate_phrases(gen_stage_count + 1)
        few_shot_examples = []

        for stage_index in range(gen_stage_count + 1):
            self.logger.info(f"Generation STAGE {stage_index}:")
            for label in tqdm(labels, desc="Generating synthetic data"):
                if train_samples is not None and label in sample_dict:
                    if stage_index == 0:
                        few_shot_examples = sample_dict[label]
                        synthetic_data.extend([{'text': ex.strip(), 'label': route_template.format(label), 'is_user_sample': True} for ex in few_shot_examples])
                    else:    
                         few_shot_examples = [synthetic_datum['text'] for synthetic_datum in synthetic_data if synthetic_datum['label'] == route_template.format(label)]

                    prompt = f"\n\nHere are some example queries for the label '{label}':\n"
                    for i, few_shot_example in enumerate(few_shot_examples, 1):
                        prompt += f"{i}. {few_shot_example}\n"
                    if label == self.oos_label:    
                        prompt += f"\nNow, generate {stage_wise_counts[stage_index]} more {stage_wise_phrases[stage_index]} short queries FAITHFUL to each of the examples above with domain terminology for the label: '{label}' in '{domain}' domain"
                    else:
                        prompt += f"\nNow, generate {stage_wise_counts[stage_index]} more {stage_wise_phrases[stage_index]} short, high-quality, realistic queries with domain terminology for the label: '{label}' in '{domain}' domain"    
                else:
                    prompt = f"\n\nGenerate minimum of {stage_wise_counts[stage_index]} {stage_wise_phrases[stage_index]} short, high-quality, realistic queries with domain terminology for the label: '{label}' in '{domain}' domain"
                prompt = prompt if user_instructions == "" else user_instructions + "\n\n" + prompt
                self.logger.info(prompt)
                try:
                    response = llm.generate(prompt)
                    if self.llm_provider == "ollama":
                        response = response.replace("```json", "")
                        response = response.replace("```", "")
                        if response.endswith('.'):
                            response = response[:-1]
                        examples = json.loads(response)
                        self.logger.debug(response)
                    else:
                        resp_obj = json.loads(response)
                        if "queries" in resp_obj:
                            examples = resp_obj["queries"]
                        else:
                            self.logger.error("LLM Prompt failed to stick to the schema") 
                            sys.exit(0)       
                        self.logger.debug(examples)
                        time.sleep(1)
                    self.logger.info(f"Total queries for  {label} - {len(examples)}")

                    synthetic_data.extend([{'text': ex.strip(), 'label': route_template.format(label), 'is_user_sample': False} for ex in examples if ex.strip()])

                    if self.add_typo_robustness:
                        synthetic_data.extend(
                                [{'text': typo, 'label': entry['label'], 'is_user_sample': entry['is_user_sample']}
                                for entry in synthetic_data
                                for typo in self._generate_natural_typo_variants(entry['text'])]
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error generating synthetic data for label '{label}': {str(e)}")

            if self.llm_provider != "ollama":  
                #Users could be in different LLM usage tiers, Don't hit tokens/min limits
                time.sleep(10)


        if make_eval_samples:
            synthetic_df = pd.DataFrame(synthetic_data)
            train_data, eval_data = self._split_synthetic_data(synthetic_df, self.min_samples)
            train_data.to_csv(train_file, index=False)
            eval_data.to_csv(eval_file, index=False)
            self.logger.info(f"Synthetic training data saved to {train_file}")
            self.logger.info(f"Synthetic evaluation data saved to {eval_file}")
        else:
            synthetic_df = pd.DataFrame(synthetic_data)
            if self.min_samples < 12:
                train_data, eval_data = self._split_synthetic_data(synthetic_df, self.min_samples)
                train_data.to_csv(train_file, index=False)
            else:
                synthetic_df.to_csv(train_file, index=False)                    
            self.logger.info(f"Synthetic training data saved to {train_file}")
        return train_file, eval_file

    def _generate_adversarial_samples(
                                    self,
                                    train_samples: Dataset,
                                    number_of_synthetic_samples: int,
                                    domain: str,
                                    route_template: str,
                                    model: str = "llama3.1",
                                    prefix: str = None,
                                    user_instructions: str = ""
                                ) -> Tuple[str, str]:
        """
        Generates adversarial samples using an LLM.
        """
        adversarial_data = []
        train_file = os.path.join(self.synthetic_data_dir, f"{prefix}_train_adversarial.csv")
        eval_file = os.path.join(self.synthetic_data_dir, f"{prefix}_eval_adversarial.csv")
        ADVERSARIAL_SYSTEM_PROMPT = """You are an expert adversarial query generator. A syntactically correct, well-formed JSON List as response like {"queries": ["Query1", "Query2", "Query3"]} is enough. Strictly no extra tokens."""
        sample_dict = {}
        for sample in train_samples:
            if sample['label'] not in sample_dict:
                sample_dict[sample['label']] = []
            sample_dict[sample['label']].append(sample['text'])
            
        llm = UnifiedLLM(self.llm_provider, model, system_prompt=ADVERSARIAL_SYSTEM_PROMPT)

        prompt = f"Below are the labels and examples for a {domain} domain:\n\n"
        for label in sample_dict.keys():
            few_shot_examples = sample_dict[label]
            prompt += f"Label: '{label}'\n"
            for i, few_shot_example in enumerate(few_shot_examples[:2], 1):
                prompt += f"  Example {i}: \"{few_shot_example}\"\n"

        prompt += f"\nNow, generate {number_of_synthetic_samples} or more adversarial i.e. lexically similar, realistic queries by pertubing or cloning the samples shown above that so they sound like they belong in one of the above labels, BUT DO NOT fit into any of the provided labels above. Do not repeat the queries. A syntactically correct, well-formed JSON List as response is enough. No explanations or footnotes of any sort are needed."                        
        
        prompt = prompt if user_instructions == "" else user_instructions + "\n\n" + prompt
        self.logger.info(prompt)
        try:
            response = llm.generate(prompt)
            if self.llm_provider == "ollama":
                response = response.replace("```json", "")
                response = response.replace("```", "")
                if response.endswith('.'):
                    response = response[:-1]
                examples = json.loads(response)["queries"]
                self.logger.debug(response)
            else:
                resp_obj = json.loads(response)
                if "queries" in resp_obj:
                    examples = resp_obj["queries"]
                else:
                    self.logger.error("LLM Response failed to stick to the schema") 
                    sys.exit(0)       
                self.logger.debug(examples)
            self.logger.info(f"Generated adversarial queries - {len(examples)}")
            adversarial_data.extend([{'text': ex.strip(), 'label': route_template.format(self.oos_label)} for ex in examples if ex.strip()])
        except Exception as e:
            self.logger.error(f"Error generating adversarial data: {str(e)}")
        adversarial_df = pd.DataFrame(adversarial_data)
        # train_data = adversarial_df.sample(n=self.min_samples, random_state=42)
        train_data = adversarial_df.sample(n=min(len(examples) - 1, number_of_synthetic_samples - 1), random_state=self.seed)
        eval_data = adversarial_df.drop(train_data.index)
        train_data.to_csv(train_file, index=False)
        eval_data.to_csv(eval_file, index=False)
        self.logger.info(f"Adversarial training data saved to {train_file}")
        self.logger.info(f"Adversarial test data saved to {eval_file}")
        return train_file, eval_file
    
    
    def _generate_additional_invalid_routes(self, prefix):

        if self.invalid_routes:
            train_file = os.path.join(self.synthetic_data_dir, f"{prefix}_train_additional_invalid.csv")
            add_invalid_routes_file = pkg_resources.resource_filename('route0x', 'data/outliers_v1.csv')
            outliers_df = pd.read_csv(add_invalid_routes_file)
            filtered_df = outliers_df[outliers_df['label'].isin(self.invalid_routes)]
            N = 2
            sampled_df = filtered_df.groupby('label').apply(lambda x: x.sample(min(N, len(x)))).reset_index(drop=True)
            final_df = pd.DataFrame({
                'text': sampled_df['text'],
                'label': [self.oos_label] * len(sampled_df)
            })
            final_df.to_csv(train_file, index=False)
        return train_file


    def _shuffle_dataset(self, dataset: Dataset) -> Dataset:
        """
        Shuffles the dataset.
        """
        return dataset.shuffle(seed=self.seed)
    
    def _get_embeddings(self, setfit_model: SetFitModel, texts: list, batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for the input texts using the SetFit model body.
        
        Args:
            texts (list): List of input texts to embed.
            batch_size (int): Batch size to process the text in chunks (default 32).
        
        Returns:
            np.ndarray: The concatenated embeddings for all input texts.
        """
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = setfit_model.model_body.encode(batch_texts, normalize_embeddings=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def _calibrate_classifer(self, output_dir, val_text_embeddings, val_labels):
        try:

            classifier_path = os.path.join(output_dir, "route0x_model", "model_head.pkl")
            classifier_head = joblib.load(classifier_path)
            self.logger.info(classifier_head.classes_)
            
            calibrated_model = CalibratedClassifierCV(
                estimator=classifier_head, 
                cv='prefit',
                method='sigmoid'
            )
            
            calibrated_model.fit(val_text_embeddings, val_labels)
            
            before_conf = classifier_head.predict_proba(val_text_embeddings[0:1]).max()
            after_conf = calibrated_model.predict_proba(val_text_embeddings[0:1]).max()
            self.logger.info(f"Sample Confidence before/after calibration: {before_conf:.3f}/{after_conf:.3f}")
            
            calibrated_path = os.path.join(output_dir, "route0x_model", "model_head_calibrated.pkl")
            joblib.dump(calibrated_model, calibrated_path)
            self.logger.info("Calibrated head saved successfully")
            self.logger.debug(calibrated_model.classes_)
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {str(e)}")
            raise

    def _train_and_evaluate(self, train_dataset, 
                           eval_dataset,
                           train_wo_adv_dataset, 
                           eval_wo_adv_dataset, 
                           model_name: str, 
                           routes, 
                           min_samples: int = 12, 
                           epochs: int = 1, 
                           max_steps: int =100, 
                           batch_size: int = 16, 
                           lr: float = 2e-5, 
                           warmup_proportion: float = 0.1, 
                           output_dir: str = "output", 
                           save_strategy: str = "no", 
                           resume_from_checkpoint: Optional[str] = None,
                           max_query_len: int=24,
                           ):
        """
        Trains and evaluates the model.
        """
        if routes:
            model = SetFitModel.from_pretrained(model_name, labels = routes, trust_remote_code=True, device=self.device)
        else:    
            model = SetFitModel.from_pretrained(model_name, trust_remote_code=True, device=self.device)

        self.logger.info("model loaded")

        if not self.only_oos_head:

            use_amp = True if torch.cuda.is_available() else False
            if max_steps == -1:
                training_args = TrainingArguments(
                    output_dir=output_dir,                
                    evaluation_strategy=save_strategy, 
                    save_strategy=save_strategy,          
                    batch_size=batch_size, 
                    num_epochs=epochs,
                    body_learning_rate = lr,         
                    warmup_proportion=warmup_proportion,   
                    logging_dir=f"{output_dir}/logs",  
                    logging_steps = 100,  
                    load_best_model_at_end=True,  
                    show_progress_bar = False,
                    use_amp = use_amp,
                    samples_per_label=min_samples,
                    loss=LOSS_FUNCTIONS[self.loss_funct_name],
                    max_length = max_query_len,
                    seed = self.seed,
                )
            else:
                training_args = TrainingArguments(
                    output_dir=output_dir,                
                    evaluation_strategy=save_strategy, 
                    save_strategy=save_strategy,          
                    batch_size=batch_size, 
                    max_steps=max_steps,
                    body_learning_rate = lr,         
                    warmup_proportion=warmup_proportion,   
                    logging_dir=f"{output_dir}/logs",    
                    load_best_model_at_end=True,  
                    logging_steps = 100,
                    show_progress_bar = False,
                    use_amp = use_amp,
                    samples_per_label=min_samples,
                    loss=LOSS_FUNCTIONS[self.loss_funct_name],
                    max_length = max_query_len,
                    seed = self.seed,
                )


            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                metric="accuracy",
                column_mapping={"text": "text", "label": "label"},
            )

            training_args.eval_strategy = training_args.evaluation_strategy
            trainer.args = training_args

            os.makedirs(output_dir, exist_ok=True)
            if resume_from_checkpoint:
                raise NotImplementedError("resume_from_checkpoint is not yet implemented.")
            else:
                trainer.train(args=training_args)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, f"run_{timestamp}")
            trainer.model.save_pretrained(os.path.join(output_dir, "route0x_model"))    
        else:
            parts = model_name.split('/', 3)    
            output_dir =  '/'.join(parts[:3])

        contamination_pct = self.expected_oos_proportion

        inlier_dataset = concatenate_datasets([train_wo_adv_dataset,eval_wo_adv_dataset])
        full_dataset = concatenate_datasets([train_dataset,eval_dataset])

        if not self.only_oos_head:
            inlier_embeddings = self._get_embeddings(trainer.model, inlier_dataset["text"])
            full_embeddings = self._get_embeddings(trainer.model, full_dataset["text"])
            val_embeddings = self._get_embeddings(trainer.model, eval_dataset["text"])
            outlier_detector = OutlierDetector(contamination=contamination_pct, 
                                            output_dir=os.path.join(output_dir, "route0x_model"), 
                                            n_neighbors=self.nn_for_oos_detection)
        else:
            inlier_embeddings = self._get_embeddings(model, inlier_dataset["text"])
            full_embeddings = self._get_embeddings(model, full_dataset["text"])
            val_embeddings = self._get_embeddings(model, eval_dataset["text"])
            outlier_detector = OutlierDetector(contamination=contamination_pct, 
                                            output_dir=os.path.join(output_dir, "route0x_model"), 
                                            n_neighbors=self.nn_for_oos_detection)

        outlier_detector.train_outliers(inlier_embeddings, inlier_dataset)
        outlier_detector.save_outliers()
        self.logger.info("All models trained and saved.")


        vectordb = VectorDB()
        index_file_path = os.path.join(output_dir, "route0x_model")
        vectordb.build_index(full_embeddings, full_dataset["label"], index_file_path)
        self.logger.info("Vectordb loaded")

        
        self._calibrate_classifer(output_dir, val_embeddings, eval_dataset["label"])

        all_metrics = {}
        if not self.skip_eval:
            all_models = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])
            for idx, model_path in enumerate(all_models, 1):
                self.logger.info(f"Evaluating model from {model_path}\n\n")
                model_dir = os.path.join(output_dir, "route0x_model")
                classifier_head = joblib.load(os.path.join(model_dir, "model_head.pkl"))
                labels = classifier_head.classes_
                metrics, report = self._evaluate_model(trainer, eval_dataset, labels)
                self._save_checkpoint_results(metrics, report, output_dir)
                all_metrics[f"model_{idx}"] = metrics

        return all_metrics, output_dir

    def _save_checkpoint_results(self, metrics, report, output_dir):
        """
        Saves checkpoint results to the output directory.
        """
        checkpoint_result_dir = os.path.join(output_dir, "eval")
        os.makedirs(checkpoint_result_dir, exist_ok=True)
        metrics_file = os.path.join(checkpoint_result_dir, "metrics.json")
        report_file = os.path.join(checkpoint_result_dir, "classification_report.txt")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        with open(report_file, 'w') as f:
            f.write(report)
        self.logger.info(f"Evaluation results are saved to {checkpoint_result_dir}")

    def _evaluate_model(self, trainer, eval_dataset, routes=None):
        """
        Evaluates the model on the test dataset.
        """
        metrics = trainer.evaluate()
        predictions = trainer.model.predict(eval_dataset['text'])
        true_labels = eval_dataset['label']
        report = classification_report(true_labels, predictions, target_names=routes)
        return metrics, report

    def _save_final_report(self, hyperparams, all_metrics, output_dir):
        """
        Saves the final report with hyperparameters and metrics.
        """
        final_report_dir = os.path.join(output_dir, "eval")
        os.makedirs(final_report_dir, exist_ok=True)
        hyperparams_file = os.path.join(final_report_dir, "hyperparameters.json")
        metrics_file = os.path.join(final_report_dir, "all_metrics.json")
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        self.logger.info(f"Final report saved to {final_report_dir}")


    def _export_onnx(self, route0x_model_path):
        """
        Exports the model to ONNX format.
        """
        route0x_model_path = os.path.join(route0x_model_path, "route0x_model")
        onnx_save_path = os.path.join(route0x_model_path, "model_body.onnx")
        quantized_model_path = os.path.join(route0x_model_path, "model_body_quantized.onnx")
        setfit_model = SetFitModel.from_pretrained(route0x_model_path, trust_remote_code=True)
        
        setfit_model = setfit_model.to(self.device)
        model_body = setfit_model.model_body
        max_seq_len = model_body.max_seq_length
        raw_model_body = model_body[0].auto_model
        dummy_input_ids = torch.randint(0, 2000, (1, max_seq_len)).to(self.device)
        dummy_attention_mask = torch.randint(0, 2, (1, max_seq_len)).to(self.device)
        dummy_token_type_ids = torch.randint(0, 2, (1, max_seq_len)).to(self.device)

        torch.onnx.export(
            raw_model_body, 
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            onnx_save_path,  
            input_names=["input_ids", "attention_mask", "token_type_ids"], 
            output_names=["sentence_embedding"],  
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"}, 
                "attention_mask": {0: "batch_size", 1: "sequence_length"}, 
                "token_type_ids": {0: "batch_size", 1: "sequence_length"}, 
                "sentence_embedding": {0: "batch_size"}  
            },
            opset_version=13
        )

        if self.do_quantise:
            self._quantize_onnx_model(onnx_save_path, quantized_model_path)

    def _quantize_onnx_model(self, onnx_model_path, quantized_model_path):
        """
        Quantizes the exported ONNX model to reduce size and improve inference speed.
        """
        quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            weight_type=QuantType.QInt8  
        )
    

    def _get_downloaded_models(self):
        """
        Retrieves the list of downloaded models.
        """
        models_info = ollama.list()
        downloaded_models = [model['name'] for model in models_info['models']]
        return downloaded_models

    def _check_and_pull_model(self, model_name: str):
        """
        Checks if a model is downloaded and pulls it if not.
        """
        downloaded_models = self._get_downloaded_models()
        if not any(model_name in downloaded_model for downloaded_model in downloaded_models):
            self.logger.info(f"Downloading model '{model_name}'...")
            ollama.pull(model_name)
            self.logger.info(f"Model '{model_name}' has been successfully downloaded.")  

    def _is_local_llm_server_running(self):
        """
        Checks if the local LLM server is running.
        """
        try:
            response = requests.get(f'{self.local_llm_host}')
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            return False

    def _detect_environment(self):
        """
        Detects the execution environment.
        """
        try:
            from google.colab import _message
            environment = 1
        except ImportError:
            environment = 0
        os_info = {
            'System': platform.system(),
            'Node Name': platform.node(),
            'Release': platform.release(),
            'Version': platform.version(),
            'Machine': platform.machine(),
            'Processor': platform.processor()
        }
        return environment    
    
    def _remove_exited_container(self, container_name):
        """
        Removes the specified container if it is exited.
        """
        try:
            output = subprocess.check_output(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"]
            )
            status = output.decode("utf-8").strip()
            if "Exited" in status:
                subprocess.run(["docker", "rm", container_name], check=True)
                self.logger.debug(f"Removed exited container: {container_name}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error removing container: {e}")


    def _start_llm_server(self):
        """
        Starts the LLM server if it's not running.
        """
        if not self._is_local_llm_server_running():
            self.logger.info("Local LLM server is not running.")
            self.logger.info("1st time setup might take a few minutes. Please be patient...")
            try:
                if self._detect_environment() == 1:
                    from IPython import get_ipython
                    self.logger.info("Attempting to install and start LLM server in Colab...")
                    get_ipython().system('curl https://ollama.ai/install.sh | sh')
                    get_ipython().system('ollama serve > rocama.log 2>&1 &')
                else:
                    self._remove_exited_container("ollama")
                    self.logger.info("Attempting to start LLM server via Docker...")
                    subprocess.run(
                        ['docker', 'run', '-d', '-v', 'ollama:/root/.ollama', '-p', '11434:11434', '--name', 'ollama', 'ollama/ollama'],
                        check=True
                    )
                time.sleep(10) 
                if self._is_local_llm_server_running():
                    self.logger.info("LLM server started successfully.")
                else:
                    self.logger.error("Failed to start Ollama server. Please check the logs for more details.")
                    if self._detect_environment() == 1:
                        self.logger.error("Ensure that Ollama was successfully installed in Colab. Check if 'ollama serve' is running.")
                    else:
                        self.logger.error("Ensure Docker is installed and that the Ollama Docker container is running.")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error while starting Ollama server via Docker: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error while starting Ollama server: {str(e)}")
        else:
            self.logger.info("LLM server is already running.")

    def _is_json_serializable(self, obj):
        """
        Check if an object is JSON serializable.
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    def _generate_natural_typo_variants(self, query, num_variants=2):
        """
        Generate naturally typo-augmented versions of a given query.
        
        Parameters:
            query (str): The original query.
            num_variants (int): Number of typo-augmented queries to generate.
            
        Returns:
            List[str]: List of typo-augmented queries.
        """
        
        augmented_queries = set()
        
        aug_keyboard = nac.KeyboardAug(min_char=1, aug_char_min=1, aug_char_max=1)       # Subtle keyboard typos
        aug_delete = nac.RandomCharAug(action="delete", aug_char_min=1, aug_char_max=1)  # Single character deletion
        aug_swap = nac.RandomCharAug(action="swap", aug_char_min=1, aug_char_max=1)      # Swap adjacent characters
        
        while len(augmented_queries) < num_variants:
            # Randomly select an augmentation type for each variant
            augmenters = [aug_keyboard, aug_delete, aug_swap]
            augmenter = random.choice(augmenters)
            
            # Apply the selected augmentation and retrieve a single string result
            typo_query = augmenter.augment(query)[0]
            
            # Ensure the typo-augmented query is distinct and add it to the set
            augmented_queries.add(typo_query)

        return list(augmented_queries)[:num_variants]



    def build_routes(self):
        """
        Builds the semantic query routes based on the provided configurations.
        """
        print(self.LOGO)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"synthetic_{self.llm_name}_{self.domain}_{timestamp}"    
        prefix = prefix.replace(" ", "_")
        self.synthetic_data_dir = "generated_datasets" 
        if not os.path.exists(self.synthetic_data_dir):
            os.makedirs(self.synthetic_data_dir, exist_ok=True)
        #PATH 0: Users want to test with a intent classification dataset with labels as routes (Only for advanced users / researchers)    
        if self.hf_dataset:
            train_dataset, eval_dataset = self._load_data(None, hf_dataset=self.hf_dataset, label_column=self.label_column, text_column=self.text_column, train_samples=self.min_samples, label_mapping=self.routes)
        elif self.enable_synth_data_gen:  

            self.llm_provider = self._get_llm_provider(self.llm_name)
            self.llm_provider = "ollama" if not self.llm_provider else self.llm_provider

            if self.llm_provider == "ollama":
                self._start_llm_server()
                self._check_and_pull_model(self.llm_name)

            if not self.domain:
                self.logger.error("Missing domain. Accurately describing your domain helps better synthetic data aug/gen")
                sys.exit(0)

            #PATH 1: Users have routes aka labels and 2 samples per route (Recommended path #1 via request obj)    
            if self.build_request:
                train_path = f"{self.synthetic_data_dir}/train_{self.domain}_{timestamp}.csv"
                self.build_request.to_csv(train_path)
                number_of_synthetic_samples = self.samples_per_route if self.samples_per_route else self.fallback_samples_per_route 
                train_dataset = self._load_data(file_path=train_path)
                self.logger.info(f"Augmenting synthetic data from your sample <query,label> pairs:")
                if self.eval_path:
                    number_of_synthetic_samples = self.samples_per_route if self.samples_per_route else self.min_samples
                    train_file, eval_file = self._generate_synthetic_queries(self.routes, self.domain, samples_per_route=number_of_synthetic_samples, model=self.llm_name, route_template=self.route_template, prefix=prefix, user_instructions=self.instruct_llm, train_samples=train_dataset, make_eval_samples=False)
                    eval_file = self.eval_path  
                else:    
                    train_file, eval_file = self._generate_synthetic_queries(self.routes, self.domain, samples_per_route=number_of_synthetic_samples, model=self.llm_name, route_template=self.route_template, prefix=prefix, user_instructions=self.instruct_llm, train_samples=train_dataset)
            if self.routes:
                number_of_synthetic_samples = self.samples_per_route if self.samples_per_route else self.fallback_samples_per_route 
                #PATH 2a: Users ONLY have list of routes aka labels (Recommended path, for non-technical users)    
                if not self.train_path and not self.eval_path:
                    self.logger.info(f"Generating synthetic data for your labels: {self.routes}")
                    train_file, eval_file = self._generate_synthetic_queries(self.routes, self.domain, samples_per_route=number_of_synthetic_samples, model=self.llm_name, route_template=self.route_template, prefix=prefix, user_instructions=self.instruct_llm)
                #PATH 2b: Users have list of routes and eval file (Only for advanced users / researchers)
                elif not self.train_path and self.eval_path:
                    self.logger.info(f"Generating synthetic data for your labels: {self.routes}")
                    train_file, eval_file = self._generate_synthetic_queries(self.routes, self.domain, samples_per_route=number_of_synthetic_samples, model=self.llm_name, route_template=self.route_template, prefix=prefix, user_instructions=self.instruct_llm)    
                    eval_file = self.eval_path  
            else:
                #PATH 3a: Users have a train.csv with 1 or more sample queries per route and full on eval.csv (Only for advanced users / researchers)
                if self.train_path and self.eval_path: 
                    self.logger.info(f"Ignoring samples_per_route param")
                    number_of_synthetic_samples = self.samples_per_route if self.samples_per_route else self.min_samples
                    train_dataset = self._load_data(self.train_path)
                    self.logger.info(f"Augmenting synthetic data from your sample <query,label> pairs:")
                    train_file, eval_file = self._generate_synthetic_queries(self.routes, self.domain, samples_per_route=number_of_synthetic_samples, model=self.llm_name, route_template=self.route_template, prefix=prefix, user_instructions=self.instruct_llm, train_samples=train_dataset, make_eval_samples=False)
                    eval_file = self.eval_path
                #PATH 3b: Users have a train.csv with 1 or more sample queries per route (Recommended path #1 via csv files)    
                elif self.train_path and not self.eval_path:
                    number_of_synthetic_samples = self.samples_per_route if self.samples_per_route else self.fallback_samples_per_route 
                    train_dataset = self._load_data(self.train_path)
                    self.logger.info(f"Augmenting synthetic data from your sample <query,label> pairs:")
                    train_file, eval_file = self._generate_synthetic_queries(self.routes, self.domain, samples_per_route=number_of_synthetic_samples, model=self.llm_name, route_template=self.route_template, prefix=prefix, user_instructions=self.instruct_llm, train_samples=train_dataset)
        else:    
            #PATH 4: Users have full-on train.csv and eval.csv (Only for advanced users / researchers)
            self.enable_id_oos_gen = False
            train_file, eval_file = self.train_path, self.eval_path

        train_without_adv_dataset = self._load_data(train_file)
        eval_without_adv_dataset =  self._load_data(eval_file)
        train_without_adv_dataset = train_without_adv_dataset.filter(lambda x: x['label'] != self.route_template.format(self.oos_label))
        eval_without_adv_dataset = eval_without_adv_dataset.filter(lambda x: x['label'] != self.route_template.format(self.oos_label))

        self.logger.info(f"Training dataset w/o ID OOS {train_without_adv_dataset.num_rows}")
        self.logger.info(f"Eval dataset w/o ID OOS {eval_without_adv_dataset.num_rows}")

        if self.enable_id_oos_gen:
            self.logger.info(f"Generating adversarial i.e. ID OOS based on training samples")
            train_adv_file, eval_adv_file = self._generate_adversarial_samples(train_samples=train_without_adv_dataset, number_of_synthetic_samples=number_of_synthetic_samples + 10, domain=self.domain, route_template=self.route_template, model=self.llm_name, prefix=prefix, user_instructions=self.instruct_llm)
            train_dataset = concatenate_datasets([train_without_adv_dataset, self._load_data(train_adv_file)])
            eval_dataset = concatenate_datasets([eval_without_adv_dataset, self._load_data(eval_adv_file)])
            train_dataset.to_csv(train_file, index=False)
            eval_dataset.to_csv(eval_file, index=False)
        else:    
            self.logger.info(f"Warning: enable_id_oos_gen is turned off, assuming your training samples have `{self.oos_label}` labeled queries")
            train_dataset = self._load_data(train_file)
            eval_dataset =  self._load_data(eval_file)

        self.logger.info(f"Training dataset w/ ID OOS {train_dataset.num_rows}")
        self.logger.info(f"Test dataset w/ ID OOS {eval_dataset.num_rows}")

        if self.add_additional_invalid_routes:    
            train_additional_invalid_file = self._generate_additional_invalid_routes(prefix)
            train_additional_invalid_dataset = self._load_data(train_additional_invalid_file)
            train_dataset = concatenate_datasets([train_dataset, train_additional_invalid_dataset])
            train_dataset.to_csv(train_file, index=False)
            self.logger.info(f"Training dataset w Additional Invalid OOS {train_dataset.num_rows}")

        if not self._assess_data(train_dataset, self.min_samples):
            self.logger.error("Your train slice doesn't have enough samples, We recommend a minimum of 12 samples per route. Add more samples or choose ask route0x to generate synthetic data")
            sys.exit(0)

        train_dataset = self._shuffle_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self._shuffle_dataset(eval_dataset)
        self.logger.info(f"Training route0x model using {self.model_name} for {self.epochs} epochs...")

        all_metrics, run_dir = self._train_and_evaluate(
            train_dataset, 
            eval_dataset, 
            train_without_adv_dataset,
            eval_without_adv_dataset,
            model_name=self.model_name, 
            epochs=self.epochs, 
            max_steps = self.max_steps if self.max_steps else -1,
            batch_size=self.batch_size, 
            lr=self.lr, 
            warmup_proportion=self.warmup_proportion,        
            min_samples=self.min_samples,
            routes = self.routes,
            max_query_len = self.max_query_len
        )

        hyperparams = {k: v for k, v in vars(self).items() if self._is_json_serializable(v)}  
        self._save_final_report(hyperparams, all_metrics, output_dir=f"{run_dir}")
        self._export_onnx(run_dir)
        self.logger.info("Model training and evaluation completed.")
        self.logger.info("Thank you for using route0x! May all your queries find their way.")


