import random
import yaml
import copy
import pdb
from typing import List, Union, Dict, Tuple
import logging
import itertools

import numpy as np
import torch
import transformers
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class LLM:

    """
    Wrapper for a language model that can be used to generate string outputs for a given input.
    """

    def __init__(self, llm_config: Dict):
        """
        Loads pre-trained large language model from Hugging Face model hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

        try:
            sample = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
                #**self.generation_params
                #streamer=streamer
            )
        except Exception as e:
            pdb.set_trace()
        output = self.tokenizer.decode(sample[0][input_ids.shape[1]:], skip_special_tokens=True)
        return output.strip()
