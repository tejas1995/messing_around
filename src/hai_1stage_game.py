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
from transformers import LogitsProcessorList

from llm import LLM
from utils import ConstrainedOutputLogitsProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

CHOICE_LETTERS = ['A', 'B', 'C', 'D']



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
        #self.name = llm_config['llm_name']
        self.name = user_config['user_name']
        self.generation_params = user_config['generation_params']
        self.instruction_prompt = user_config['instruction_prompts']['1stage_game']

        logits_processor = ConstrainedOutputLogitsProcessor(self.llm.tokenizer, valid_outputs=['yes', 'no', 'Yes', 'No'])
        self.logits_processor_list = LogitsProcessorList([logits_processor])
        logger.info(f"Loaded (simulated) user: {self.name}")
        logger.info("-"*50)
        pass


    def make_decision_with_assistant_help(
        self, 
        task_inputs: Dict,
        assistance: Dict,
        assistant: Assistant
    ) -> str:
        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, task_inputs['choices'])])
        input_string = self.instruction_prompt.replace("QUESTION", task_inputs['question']).replace("CHOICES", choices_string)
        input_string = input_string.replace("ASSISTANT_ACCURACY", str(int(assistant.assistant_correctness.params['p']*100)))
        input_string = input_string.replace("ASSISTANT_PREDICTION", assistance['predicted_answer']).replace("ASSISTANT_CONFIDENCE", f"{assistance['confidence']:.0%}")

        output = self.llm.generate(input_string, max_new_tokens=1, logits_processor=self.logits_processor_list)
        #output = self.llm.generate(input_string, **self.generation_params)
        answer = output.split('\n')[0].strip().lower()

        try:
            assert answer.startswith('yes') or answer.startswith('no')
        except:
            print(output)
            pdb.set_trace()
        #print("INPUT:")
        #print(input_string)
        #print("\nLLM OUTPUT:")
        #print(output)
        #pdb.set_trace()
        return 1 if 'yes' in answer else 0, output

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_config", type=str, default='configs/game.yaml')
    parser.add_argument("--assistant_config", type=str, default='configs/')
    parser.add_argument("--output_dir", type=str, default='/home/tejas/experiments/hai_game/1stage_game/')
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
        #if idx >= 50:
        #    break
        try:
            assistant_pred = assistant.get_answer(task_inputs=d)
            user_agrees_with_assistant, user_output = user.make_decision_with_assistant_help(
                task_inputs=d,
                assistance=assistant_pred,
                assistant=assistant
            )
        except Exception as e:
            print(e)
            pdb.set_trace()
            continue
        true_answer = CHOICE_LETTERS[d['answer']]

        results.append({
            "idx": idx,
            "task_inputs": d,
            "true_answer": true_answer,
            "assistant_pred": assistant_pred['predicted_answer'],
            "assistant_conf": assistant_pred['confidence'],
            "assistant_is_correct": assistant_pred['is_correct'],
            "user_agrees_with_assistant": user_agrees_with_assistant,
            "user_output": user_output
        })

        t.set_description(f"Attempted: {idx+1}, Completed: {len(results)}")

    ece = calculate_ece(results)
    logger.info(f"Assistant Calibration Error: {ece:.4f}")
    logger.info("-"*100)
    
    # Assistant accuracy: % instances where Assistant is correct
    assistant_accuracy = sum([r['assistant_is_correct'] for r in results]) / len(results)
    logger.info(f"Assistant accuracy: {assistant_accuracy:.2%}")

    # User reliance: % instances where user agrees with Assistant
    user_reliance = sum([r['user_agrees_with_assistant'] for r in results]) / len(results)
    logger.info(f"User reliance: {user_reliance:.2%}")
    logger.info("-"*100)

    # Label: is the Assistant correct? (i.e. should the user rely on the Assistant?)
    # Prediction: does the user agree with the Assistant?
    # TP: Assistant is correct and user agrees with Assistant
    tp = sum([r['assistant_is_correct'] and r['user_agrees_with_assistant'] for r in results])
    # FP: Assistant is incorrect and user agrees with Assistant
    fp = sum([not r['assistant_is_correct'] and r['user_agrees_with_assistant'] for r in results])
    # TN: Assistant is incorrect and user disagrees with Assistant
    tn = sum([not r['assistant_is_correct'] and not r['user_agrees_with_assistant'] for r in results])
    # FN: Assistant is correct and user disagrees with Assistant
    fn = sum([r['assistant_is_correct'] and not r['user_agrees_with_assistant'] for r in results])
    logger.info(f"Instances of user correctly relying on Assistant suggestion: {tp}")
    logger.info(f"Instances of user incorrectly relying on Assistant suggestion: {fp}")
    logger.info(f"Instances of user correctly not relying on Assistant suggestion: {tn}")
    logger.info(f"Instances of user incorrectly not relying on Assistant suggestion: {fn}")
    logger.info("-"*100)

    # Recall: tp/(tp+fn) = out of trials where Assistant is correct, % of times user agrees with Assistant
    recall = tp / (tp + fn)
    logger.info(f"Recall: {recall:.2%}")

    # Precision: tp/(tp+fp) = out of trials where user agrees with Assistant, % of times Assistant is correct
    precision = tp / (tp + fp)
    logger.info(f"Precision: {precision:.2%}")

    f1 = 2 * (precision * recall) / (precision + recall)
    logger.info(f"F1 Score: {f1:.2f}")

    # FPR: fp/(fp+tn) = out of trials where Assistant is incorrect, % of times user agrees with Assistant
    fpr = fp / (fp + tn)
    logger.info(f"False Positive Rate: {fpr:.2%}")

    results_dict = {
        "experiment_name": experiment_name,
        "assistant_calibration_error": ece,
        "assistant_accuracy": assistant_accuracy,
        "user_reliance": user_reliance,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "rollout_results": results,
    }
    json.dump(results_dict, open(output_file, 'w'), indent=2)
    logger.info(f"Results saved to {output_file}")
    #pdb.set_trace()

if __name__ == '__main__':
    main()