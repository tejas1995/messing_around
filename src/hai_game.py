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

import scipy
import numpy as np
import datasets

from llm import LLM

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

CHOICE_LETTERS = ['A', 'B', 'C', 'D']

def calculate_flip_rate(results: List[Dict]) -> float:
    instances_where_ai_and_human_initial_guess_differ = [r for r in results if not r['initial_guesses_match']]
    flip_rate = sum([r['human_flipped'] for r in instances_where_ai_and_human_initial_guess_differ]) / len(instances_where_ai_and_human_initial_guess_differ)
    return flip_rate, len(instances_where_ai_and_human_initial_guess_differ)

def calculate_switch_percentage(results: List[Dict]) -> float:
    instances_where_ai_and_human_initial_guess_differ = [r for r in results if not r['initial_guesses_match']]
    switch_percentage = sum([r['final_guess_agrees_with_ai'] for r in instances_where_ai_and_human_initial_guess_differ]) / len(instances_where_ai_and_human_initial_guess_differ)
    return switch_percentage, len(instances_where_ai_and_human_initial_guess_differ)

def calculate_appropriate_reliance(results: List[Dict]) -> float:
    reliance_score = 0.0
    for r in results:
        if r['ai_is_correct'] and r['final_guess_agrees_with_ai']:
            reliance_score += 1
        elif not r['ai_is_correct'] and not r['final_guess_agrees_with_ai']:
            reliance_score += 1
    return reliance_score / len(results)

def calculate_ece(calibration_samples: List[Dict], num_bins=10) -> float:
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    confidences = np.array([x['confidence'] for x in calibration_samples])
    accuracies = np.array([x['is_correct'] for x in calibration_samples])
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
        assistant_name = f"{assistant_config['assistant_correctness']['params']['p']}accuracy"
        assistant_name += f"-correctconf_{assistant_config['confidence_when_correct']['distribution_name']}"
        for k, v in assistant_config['confidence_when_correct']['params'].items():
            assistant_name += f"_{v}{k}"
        assistant_name += f"-incorrectconf_{assistant_config['confidence_when_incorrect']['distribution_name']}"
        for k, v in assistant_config['confidence_when_incorrect']['params'].items():
            assistant_name += f"_{v}{k}"
        self.name = assistant_name
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

class Human:
    def __init__(self, human_config: Dict) -> None:
        llm_config = yaml.safe_load(open(human_config['llm_config_path']))
        self.llm = LLM(llm_config=llm_config)
        self.name = llm_config['llm_name']
        logger.info(f"Loaded (simulated) human: {self.name}")
        logger.info("-"*50)
        pass

    def make_initial_decision(
        self, 
        task_inputs: Dict,
    ) -> str:
        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, task_inputs['choices'])])
        input_string = "You are trying to answer a question. " \
            + " You have to answer the question based on your own knowledge. \n\n" \
            + f"Question: {task_inputs['question']} \nChoices: {choices_string} \n" \
            + "What is your final answer (A, B, C or D)?\nAnswer:"
        output = self.llm.generate(input_string)
        answer = output.split('\n')[0].split('.')[0].strip()
        #pdb.set_trace()
        assert answer in CHOICE_LETTERS
        return answer

    def make_decision_with_ai_assistance(
        self, 
        task_inputs: Dict,
        initial_decision: str,
        ai_assistance: Dict
    ) -> str:
        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, task_inputs['choices'])])
        input_string = "You are trying to answer a question. " \
            + " You also have an AI assistant to help you answer this question. The AI assistant is not always confident in its answers. " \
            + " You have to answer the question based on your own knowledge, as well as the AI assistant's suggestion and confidence. \n\n" \
            + f"Question: {task_inputs['question']} \nChoices: {choices_string}\n\n" \
            + f"Your initial answer: {initial_decision} \n\n" \
            + f"AI assistant's answer: {ai_assistance['predicted_answer']} \nAI assistant's confidence: {ai_assistance['confidence']:.0%} \n\n" \
            + "What is your final answer (A, B, C or D)?\nAnswer:"
        output = self.llm.generate(input_string)
        answer = output.split('\n')[0].split('.')[0].strip()
        assert answer in CHOICE_LETTERS
        #pdb.set_trace()
        return answer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='configs/game.yaml')
    parser.add_argument("--output_dir", type=str, default='/home/tejas/experiments/hai_game/')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file))
    human = Human(config['human'])
    assistant = Assistant(config['assistant'])

    dataset = datasets.load_dataset("lighteval/mmlu", 'clinical_knowledge', split='test')

    experiment_name = f"dataset-mmlu_clinical/simulated_human-{human.name}/assistant-{assistant.name}"
    output_file = os.path.join(args.output_dir, f"{experiment_name}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Output file: {output_file}")

    # Compute assistant calibration error
    calibration_samples = []
    for i in tqdm(range(1000), desc="Computing assistant calibration error..."):
        ai_pred = assistant.get_answer(task_inputs={})
        calibration_samples.append(ai_pred)
    ece = calculate_ece(calibration_samples)
    logger.info(f"Assistant Calibration Error: {ece:.4f}")
    #pdb.set_trace()

    # Run the H-AI interaction game
    t = tqdm(dataset)
    results = []
    for idx, d in enumerate(t):
        try:
            ai_pred = assistant.get_answer(task_inputs=d)
            human_initial_pred = human.make_initial_decision(
                task_inputs=d
            )
            human_final_pred = human.make_decision_with_ai_assistance(
                task_inputs=d,
                initial_decision=human_initial_pred,
                ai_assistance=ai_pred
            )
        except Exception as e:
            #print(e)
            continue
        true_answer = CHOICE_LETTERS[d['answer']]
        human_is_correct = human_final_pred == true_answer
        human_flipped = human_initial_pred != human_final_pred
        initial_guesses_match = human_initial_pred == ai_pred['predicted_answer']
        final_guess_agrees_with_ai = human_final_pred == ai_pred['predicted_answer']
        initial_guess_correct = human_initial_pred == true_answer

        results.append({
            "idx": idx,
            "task_inputs": d,
            "true_answer": true_answer,
            "human_initial_pred": human_initial_pred,
            "human_final_pred": human_final_pred,
            "ai_pred": ai_pred['predicted_answer'],
            "ai_conf": ai_pred['confidence'],
            "ai_is_correct": ai_pred['is_correct'],
            "human_is_correct": human_is_correct,
            "human_flipped": human_flipped,
            "initial_guesses_match": initial_guesses_match,
            "initial_guess_correct": initial_guess_correct,
            "final_guess_agrees_with_ai": final_guess_agrees_with_ai,
        })

        t.set_description(f"Attempted: {idx+1}, Completed: {len(results)}")
    
    # Initial guess accuracy: % instances where human initial guess is correct
    initial_human_accuracy = sum([r['initial_guess_correct'] for r in results]) / len(results)
    logger.info(f"Initial guess accuracy of humans: {initial_human_accuracy:.2%} (out of {len(results)} instances)")

    # AI accuracy: % instances where AI is correct
    assistant_accuracy = sum([r['ai_is_correct'] for r in results]) / len(results)
    logger.info(f"AI accuracy: {assistant_accuracy:.2%} (out of {len(results)} instances)")

    # Human final accuracy: % instances where human final guess is correct
    final_human_accuracy = sum([r['human_is_correct'] for r in results]) / len(results)
    logger.info(f"Final guess accuracy of humans: {final_human_accuracy:.2%} (out of {len(results)} instances)")

    logger.info("-"*100)

    # Flip rate: % of trials in which final human guess ≠ initial guess, out of trials where AI prediction ≠ initial human guess
    flip_rate, num_instances = calculate_flip_rate(results)
    logger.info(f"Flip rate: {flip_rate:.2%} (out of {num_instances} instances where AI and human initial guess differ)")

    # Switch percentage: % of trials in which final human guess ==  AI prediction, out of trials where AI prediction ≠ initial human guess
    switch_percentage, num_instances = calculate_switch_percentage(results)
    logger.info(f"Switch percentage: {switch_percentage:.2%} (out of {num_instances} instances where AI and human initial guess differ)")

    # Agreement percentage: % of trials in which the participant’s final prediction agreed with the AI’s prediction (out of all trials)
    agreement_percentage = sum([r['final_guess_agrees_with_ai'] for r in results]) / len(results)
    logger.info(f"Agreement percentage: {agreement_percentage:.2%} (out of {len(results)} instances)")

    # Appropriate reliance: % of trials where AI is correct and human final guess agrees with AI, 
    #                       or AI is incorrect and human final guess disagrees with AI
    appropriate_reliance = calculate_appropriate_reliance(results)
    logger.info(f"Appropriate reliance: {appropriate_reliance:.2%} (out of {len(results)} instances)")

    results_dict = {
        "experiment_name": experiment_name,
        "initial_human_accuracy": initial_human_accuracy,
        "assistant_accuracy": assistant_accuracy,
        "final_human_accuracy": final_human_accuracy,
        "flip_rate": flip_rate,
        "switch_percentage": switch_percentage,
        "agreement_percentage": agreement_percentage,
        "appropriate_reliance": appropriate_reliance,
        "assistant_calibration_error": ece,
        "rollout_results": results,
    }
    json.dump(results_dict, open(output_file, 'w'), indent=2)
    logger.info(f"Results saved to {output_file}")
    #pdb.set_trace()

if __name__ == '__main__':
    main()