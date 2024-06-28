import os, json
import yaml
import copy
import pdb
from typing import List, Union, Dict, Tuple
import logging
import itertools
import argparse
from tqdm import tqdm

import numpy as np
import torch
import transformers
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from transformers import LogitsProcessorList
from utils import ConstrainedOutputLogitsProcessor

from data import DATASET_CLASS_MAP

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

CHOICES = ['A', 'B', 'C', 'D']

class LLMAssistant:

    """
    Wrapper for a language model that can be used to generate string outputs for a given input.
    """

    def __init__(self, llm_assistant_config: Dict):
        """
        Loads pre-trained large language model from Hugging Face model hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        llm_config_path = llm_assistant_config['llm_config_path']
        llm_config = yaml.safe_load(open(llm_config_path, 'r'))
        self.logits_processor_list = None
        self.answer_prediction_prompt = llm_assistant_config['answer_prediction_prompt']
        self.confidence_method = llm_assistant_config['confidence_method']
        self.name = llm_assistant_config['assistant_name']

        self.load_llm(llm_config)


    def load_llm(self, llm_config: Dict):
        model_load_config = llm_config['model_load_config']
        self.use_chat_template = llm_config['use_chat_template']
        #self.generation_params = llm_config['generation_params']

        if model_load_config['load_in_4bit']:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_load_config['model_name_or_path'],
                #load_dir,
                torch_dtype=torch.bfloat16 if model_load_config['use_bfloat'] else torch.float16,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16 if model_load_config['use_bfloat'] else torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4'),
                )
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
            #print(f"Loaded model from {load_dir}")
        else:
            if model_load_config['use_bfloat']:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(model_load_config['model_name_or_path'], torch_dtype=torch.bfloat16).to(self.device)
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(model_load_config['model_name_or_path']).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_load_config['tokenizer_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()

        if model_load_config['model_name_or_path'].startswith('/'):
            logger.info(f"Loaded CausalLM model from {model_load_config['model_name_or_path']}")
        else:
            logger.info(f"Loaded pre-trained CausalLM model: {model_load_config['model_name_or_path']}")

        logits_processor = ConstrainedOutputLogitsProcessor(self.tokenizer, valid_outputs=CHOICES)
        self.logits_processor_list = LogitsProcessorList([logits_processor])


    def generate(
        self, 
        input_string: str,
        **kwargs
    ) -> str:
        """
        Generates a natural language output, given an input string.
        """
        if self.use_chat_template:
            messages = [{"role": "user", "content": input_string}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            attention_mask = torch.ones(input_ids.shape).to(self.device)
            #pdb.set_trace()
        else:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids.to(self.device)
            attention_mask = torch.ones(input_ids.shape).to(self.device)

        #try:
        sample = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
            #**self.generation_params
            #streamer=streamer
        )
        #except Exception as err:
        #    pdb.set_trace()
        output = self.tokenizer.decode(sample.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        return output.strip(), sample, input_ids.shape

    def get_answer(
        self, 
        task_inputs: Dict
    ) -> Tuple[str, float]:
        """
        Given a question and a list of answer choices, returns the most likely answer choice.
        """
        input_string = self.answer_prediction_prompt.replace('QUESTION', task_inputs['question'])
        input_string = input_string.replace('CHOICES', task_inputs['choices_string'])
        predicted_answer, sample, input_shape = self.generate(input_string, max_new_tokens=1, do_sample=False, logits_processor = self.logits_processor_list)
        token_probs = torch.nn.functional.softmax(sample.scores[0], dim=-1).squeeze().cpu().detach().numpy()
        choice_token_ids = self.tokenizer.convert_tokens_to_ids(CHOICES)
        choice_probs = [token_probs[c] for c in choice_token_ids]
        if sum(choice_probs) < 0.95:
            logger.warning(f"Sum of choice probs is less than 0.95: {sum(choice_probs)}")

        if self.confidence_method == 'token_prob':
            predicted_answer_idx = CHOICES.index(predicted_answer)
            predicted_answer_conf = choice_probs[predicted_answer_idx]
            if predicted_answer_conf < 0.25:
                pdb.set_trace()
        else:
            raise NotImplementedError

        return predicted_answer, predicted_answer_conf.item()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CLASS_MAP.keys()))
    parser.add_argument("--output_dir", type=str, default='/home/tejas/experiments/FAFO/llm_task_performance/')
    args = parser.parse_args()

    llm_assistant_config_path = 'configs/assistant_configs/llm/llama3/token_prob.yaml'
    llm_assistant_config = yaml.safe_load(open(llm_assistant_config_path, 'r'))
    llm_assistant = LLMAssistant(llm_assistant_config)
    dataset_class = DATASET_CLASS_MAP[args.dataset]
    dataset = dataset_class()

    t = tqdm(dataset)
    results = []
    for idx, d in enumerate(t):
        predicted_answer, predicted_answer_conf = llm_assistant.get_answer(d)
        results.append(
            {
                "idx": idx,
                "task_inputs": d,
                "true_answer": d['correct_answer'],
                "predicted_answer": predicted_answer,
                "correct": predicted_answer == d['correct_answer'],
                "prediction_conf": predicted_answer_conf
            }
        )

    experiment_name = f"dataset-{args.dataset}/{llm_assistant.name}"
    output_file = os.path.join(args.output_dir, f"{experiment_name}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    pdb.set_trace()