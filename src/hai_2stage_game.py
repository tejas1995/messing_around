import os
import random
import yaml
import copy
import json
import pdb
from typing import List, Union, Dict, Tuple, Any
import logging
import argparse
from tqdm import tqdm

import numpy as np
import datasets

from llm import LLM
import torch
from transformers import LogitsProcessor, LogitsProcessorList

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

CHOICE_LETTERS = ['A', 'B', 'C', 'D']


# Define a custom LogitsProcessor
class OnlyABCDLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.valid_token_ids = [
            tokenizer.convert_tokens_to_ids(c) for c in CHOICE_LETTERS
        ]

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores) * float('-inf')
        for valid_token_id in self.valid_token_ids:
            mask[:, valid_token_id] = scores[:, valid_token_id]
        return mask

def calculate_flip_rate(results: List[Dict]) -> float:
    instances_where_assistant_and_user_initial_guess_differ = [r for r in results if not r['initial_guesses_match']]
    flip_rate = sum([r['user_flipped'] for r in instances_where_assistant_and_user_initial_guess_differ]) / len(instances_where_assistant_and_user_initial_guess_differ)
    return flip_rate, len(instances_where_assistant_and_user_initial_guess_differ)

def calculate_switch_percentage(results: List[Dict]) -> float:
    instances_where_assistant_and_user_initial_guess_differ = [r for r in results if not r['initial_guesses_match']]
    switch_percentage = sum([r['final_guess_agrees_with_assistant'] for r in instances_where_assistant_and_user_initial_guess_differ]) / len(instances_where_assistant_and_user_initial_guess_differ)
    return switch_percentage, len(instances_where_assistant_and_user_initial_guess_differ)

#def calculate_appropriate_reliance(results: List[Dict]) -> float:
#    reliance_score = 0.0
#    for r in results:
#        if r['assistant_is_correct'] and r['final_guess_agrees_with_assistant']:
#            reliance_score += 1
#        elif not r['assistant_is_correct'] and not r['final_guess_agrees_with_assistant']:
#            reliance_score += 1
#    return reliance_score / len(results)

def calculate_inappropriate_reliance(results) -> float:
    filtered_results = [r for r in results if not r['assistant_is_correct']]
    num_relied = len([r for r in filtered_results if r['final_guess_agrees_with_assistant']])
    return num_relied / len(filtered_results), len(filtered_results)

def calculate_appropriate_reliance(results) -> float:
    filtered_results = [r for r in results if r['assistant_is_correct']]
    num_relied = len([r for r in filtered_results if r['final_guess_agrees_with_assistant']])
    return num_relied / len(filtered_results), len(filtered_results)

def calculate_ece(calibration_samples: List[Dict], num_bins=10) -> float:
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    confidences = np.array([x['assistant_conf'] for x in calibration_samples])
    accuracies = np.array([x['assistant_is_correct'] for x in calibration_samples])
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        #pdb.set_trace()
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


class ProbabilityDistribution:

    def __init__(self, prob_distribution_config: Dict) -> None:
        self.distribution_name = prob_distribution_config['distribution_name']
        self.params = prob_distribution_config['params']

    def draw_from_distribution(self) -> Any:
        if self.distribution_name == 'uniform':
            return np.random.uniform(low=self.params['low'], high=self.params['high'])
        elif self.distribution_name == 'normal':
            return np.random.normal(loc=self.params['mean'], scale=self.params['stdev'])
        elif self.distribution_name == 'bernoulli':
            return np.random.binomial(n=1, p=self.params['p'])
        elif self.distribution_name == 'beta':
            return np.random.beta(a=self.params['a'], b=self.params['b'])
        else:
            raise ValueError(f"Unknown distribution name: {self.distribution_name}")

class Assistant:
    def __init__(self, assistant_config: Dict) -> None:
        assert assistant_config['assistant_correctness']['distribution_name'] == 'bernoulli'    # Must be from discrete distribution
        self.assistant_correctness = ProbabilityDistribution(
            prob_distribution_config=assistant_config['assistant_correctness']
        )
        self.confidence_when_correct = ProbabilityDistribution(
            prob_distribution_config=assistant_config['confidence_when_correct']
        )
        self.confidence_when_incorrect = ProbabilityDistribution(
            prob_distribution_config=assistant_config['confidence_when_incorrect']
        )
        #assistant_name = f"{assistant_config['assistant_correctness']['params']['p']}accuracy"
        #assistant_name += f"-correctconf_{assistant_config['confidence_when_correct']['distribution_name']}"
        #if assistant_config['confidence_when_correct']['distribution_name'] not in ['uniform', 'normal']:
        #    for k, v in assistant_config['confidence_when_correct']['params'].items():
        #        assistant_name += f"_{v}{k}"
        #assistant_name += f"-incorrectconf_{assistant_config['confidence_when_incorrect']['distribution_name']}"
        #if assistant_config['confidence_when_incorrect']['distribution_name'] not in ['uniform', 'normal']:
        #    for k, v in assistant_config['confidence_when_incorrect']['params'].items():
        #        assistant_name += f"_{v}{k}"
        self.name = assistant_config['assistant_name']
        logger.info(f"Loaded assistant: {self.name}")
        logger.info("-"*50)

    def get_answer(self, task_inputs: Dict) -> str:
        is_correct = self.assistant_correctness.draw_from_distribution()
        if task_inputs != {}:
            answer_idx = task_inputs['answer'] if is_correct else random.choice([i for i in range(len(task_inputs['choices'])) if i != task_inputs['answer']])
            answer = CHOICE_LETTERS[answer_idx]
        else:
            answer = random.choice(CHOICE_LETTERS)            
        confidence = self.confidence_when_correct.draw_from_distribution() if is_correct \
            else self.confidence_when_incorrect.draw_from_distribution()
        return {"predicted_answer": answer, "confidence": confidence, "is_correct": is_correct}

class User:
    def __init__(self, user_config: Dict) -> None:
        llm_config = yaml.safe_load(open(user_config['llm_config_path']))
        self.llm = LLM(llm_config=llm_config)
        #pdb.set_trace()
        self.name = llm_config['llm_name']
        self.generation_params = user_config['generation_params']
        self.instruction_prompt = user_config['instruction_prompts']['2stage_game']
        logger.info(f"Loaded (simulated) user: {self.name}")
        logger.info("-"*50)

        logits_processor = OnlyABCDLogitsProcessor(self.llm.tokenizer)
        self.logits_processor_list = LogitsProcessorList([logits_processor])
        #pdb.set_trace()
        pass

    def make_initial_decision(
        self, 
        task_inputs: Dict,
    ) -> str:
        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, task_inputs['choices'])])
        input_string = self.instruction_prompt['initial_guess'].replace("QUESTION", task_inputs['question']).replace("CHOICES", choices_string)
        output = self.llm.generate(input_string, **self.generation_params)
        answer = output.split('\n')[0].split('.')[0].strip()[0]
        #pdb.set_trace()
        answer_extraction_input = f"Look at the following sentence: '{output}'\n\nAnswer, using one letter only, which out of A, B, C, or D they decided on."
        answer = self.llm.generate(answer_extraction_input, max_new_tokens=1, logits_processor=self.logits_processor_list)
        assert answer in CHOICE_LETTERS
        #try:
        #    assert answer in CHOICE_LETTERS
        #except:
        #    print(f"Output: {output}")
        #    pdb.set_trace()
        return answer

    def make_decision_with_assistant_help(
        self, 
        task_inputs: Dict,
        initial_decision: str,
        assistance: Dict
    ) -> str:
        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, task_inputs['choices'])])
        input_string = self.instruction_prompt['final_decision'].replace("QUESTION", task_inputs['question']).replace("CHOICES", choices_string)
        input_string = input_string.replace("INITIAL_DECISION", initial_decision)
        input_string = input_string.replace("ASSISTANT_PREDICTION", assistance['predicted_answer']).replace("ASSISTANT_CONFIDENCE", f"{assistance['confidence']:.0%}")

        generation_params = self.generation_params.copy()
        generation_params['max_new_tokens'] = 50
        output = self.llm.generate(input_string, **generation_params)
        
        answer_extraction_input = f"Look at the following sentence: '{output}'\n\nAnswer, using one letter only, which out of A, B, C, or D they decided on."
        answer = self.llm.generate(answer_extraction_input, max_new_tokens=1, logits_processor=self.logits_processor_list)
        #pdb.set_trace()
        assert answer in CHOICE_LETTERS
        return answer, output

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_config", type=str, default='configs/game.yaml')
    parser.add_argument("--assistant_config", type=str, default='configs/')
    parser.add_argument("--output_dir", type=str, default='/home/tejas/experiments/hai_game/2stage_game/')
    args = parser.parse_args()

    user_config = yaml.safe_load(open(args.user_config))
    user = User(user_config)
    assistant_config = yaml.safe_load(open(args.assistant_config))
    assistant = Assistant(assistant_config)

    dataset = datasets.load_dataset("lighteval/mmlu", 'clinical_knowledge', split='test')

    experiment_name = f"dataset-mmlu_clinical/simulated_user-{user.name}/assistant-{assistant.name}"
    output_file = os.path.join(args.output_dir, f"{experiment_name}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Output file: {output_file}")

    # Compute assistant calibration error
    #calibration_samples = []
    #for i in tqdm(range(1000), desc="Computing assistant calibration error..."):
    #    ai_pred = assistant.get_answer(task_inputs={})
    #    calibration_samples.append(ai_pred)
    #ece = calculate_ece(calibration_samples)
    #logger.info(f"Assistant Calibration Error: {ece:.4f}")
    #pdb.set_trace()

    # Run the H-AI interaction game
    t = tqdm(dataset)
    results = []
    for idx, d in enumerate(t):
        try:
            assistant_pred = assistant.get_answer(task_inputs=d)
            user_initial_pred = user.make_initial_decision(
                task_inputs=d
            )
            user_final_pred, user_final_output = user.make_decision_with_assistant_help(
                task_inputs=d,
                initial_decision=user_initial_pred,
                assistance=assistant_pred
            )
        except Exception as e:
            #print(e)
            continue
        true_answer = CHOICE_LETTERS[d['answer']]
        user_is_correct = user_final_pred == true_answer
        user_flipped = user_initial_pred != user_final_pred
        initial_guesses_match = user_initial_pred == assistant_pred['predicted_answer']
        final_guess_agrees_with_assistant = user_final_pred == assistant_pred['predicted_answer']
        initial_guess_correct = user_initial_pred == true_answer

        results.append({
            "idx": idx,
            "task_inputs": d,
            "true_answer": true_answer,
            "user_initial_pred": user_initial_pred,
            "user_final_pred": user_final_pred,
            "user_final_output": user_final_output,
            "assistant_pred": assistant_pred['predicted_answer'],
            "assistant_conf": assistant_pred['confidence'],
            "assistant_is_correct": assistant_pred['is_correct'],
            "user_is_correct": user_is_correct,
            "user_flipped": user_flipped,
            "initial_guesses_match": initial_guesses_match,
            "initial_guess_correct": initial_guess_correct,
            "final_guess_agrees_with_assistant": final_guess_agrees_with_assistant,
        })

        t.set_description(f"Attempted: {idx+1}, Completed: {len(results)}")

    ece = calculate_ece(results)
    logger.info(f"Assistant Calibration Error: {ece:.4f}")
    logger.info("-"*100)
    
    # Initial guess accuracy: % instances where user initial guess is correct
    initial_user_accuracy = sum([r['initial_guess_correct'] for r in results]) / len(results)
    logger.info(f"Initial guess accuracy of users: {initial_user_accuracy:.2%} (out of {len(results)} instances)")

    # AI accuracy: % instances where AI is correct
    assistant_accuracy = sum([r['assistant_is_correct'] for r in results]) / len(results)
    logger.info(f"AI accuracy: {assistant_accuracy:.2%} (out of {len(results)} instances)")

    # User final accuracy: % instances where user final guess is correct
    final_user_accuracy = sum([r['user_is_correct'] for r in results]) / len(results)
    logger.info(f"Final guess accuracy of users: {final_user_accuracy:.2%} (out of {len(results)} instances)")

    logger.info("-"*100)

    # Flip rate: % of trials in which final user guess ≠ initial guess, out of trials where AI prediction ≠ initial user guess
    flip_rate, num_instances = calculate_flip_rate(results)
    logger.info(f"Flip rate: {flip_rate:.2%} (out of {num_instances} instances where AI and user initial guess differ)")

    # Switch percentage: % of trials in which final user guess ==  AI prediction, out of trials where AI prediction ≠ initial user guess
    switch_percentage, num_instances = calculate_switch_percentage(results)
    logger.info(f"Switch percentage: {switch_percentage:.2%} (out of {num_instances} instances where AI and user initial guess differ)")

    # Agreement percentage: % of trials in which the participant’s final prediction agreed with the AI’s prediction (out of all trials)
    agreement_percentage = sum([r['final_guess_agrees_with_assistant'] for r in results]) / len(results)
    logger.info(f"Agreement percentage: {agreement_percentage:.2%} (out of {len(results)} instances)")

    # Appropriate reliance (recall): % of trials where AI is correct and user final guess agrees with AI, 
    appropriate_reliance, num_assistant_correct = calculate_appropriate_reliance(results)
    logger.info(f"Appropriate reliance: {appropriate_reliance:.2%} (out of {num_assistant_correct} instances)")

    # Inappropriate reliance (FPR): % of trials where AI is incorrect and user final guess agrees with assistant
    inappropriate_reliance, num_assistant_incorrect = calculate_inappropriate_reliance(results)
    logger.info(f"Inappropriate reliance: {inappropriate_reliance:.2%} (out of {num_assistant_incorrect} instances)")

    results_dict = {
        "experiment_name": experiment_name,
        "initial_user_accuracy": initial_user_accuracy,
        "assistant_accuracy": assistant_accuracy,
        "final_user_accuracy": final_user_accuracy,
        "flip_rate": flip_rate,
        "switch_percentage": switch_percentage,
        "agreement_percentage": agreement_percentage,
        "appropriate_reliance": appropriate_reliance,
        "inappropriate_reliance": inappropriate_reliance, 
        "assistant_calibration_error": ece,
        "rollout_results": results,
    }
    json.dump(results_dict, open(output_file, 'w'), indent=2)
    logger.info(f"Results saved to {output_file}")
    #pdb.set_trace()

if __name__ == '__main__':
    main()