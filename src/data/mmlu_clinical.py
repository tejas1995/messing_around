import datasets
import logging
import pdb

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

CHOICE_LETTERS = ['A', 'B', 'C', 'D']

class MMLUClinical:

    def __init__(self, split='test'):
        self.dataset = datasets.load_dataset("lighteval/mmlu", 'clinical_knowledge', split=split)
        logger.info(f"Loaded MMLU-ClinicalKnowledge dataset ({split} split, {len(self.dataset)} examples)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        correct_answer = CHOICE_LETTERS[d['answer']]
        incorrect_answers = [c for i, c in enumerate(CHOICE_LETTERS) if i != d['answer']]
        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, d['choices'])])
        d['correct_answer'] = correct_answer
        d['incorrect_answers'] = incorrect_answers
        d['choices_string'] = choices_string
        return d

if __name__ == '__main__':
    dataset = MMLUClinical()
    pdb.set_trace()