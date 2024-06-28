import pdb
import random
import datasets
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

CHOICE_LETTERS = ['A', 'B', 'C', 'D']
ALLOWED_CATEGORIES = [
    'Misinformation', 
    #'Mandela Effect', 
    #'Distraction', 
    #'Misquotations', 
    #'Confusion: Places', 
    'Psychology', 
    'Economics', 
    #'Indexical Error: Location', 
    'Education', 
    'Health', 
    'Stereotypes', 
    #'Myths and Fairytales', 
    #'Misconceptions', 
    'Sociology', 
    'Weather', 
    #'Conspiracies', 
    'Nutrition', 
    #'Confusion: People', 
    #'Statistics', 
    #'Confusion: Other', 
    'Finance', 
    #'Superstitions', 
    #'Logical Falsehood', 
    #'Proverbs', 
    #'Language', 
    'Law', 
    'History', 
    #'Fiction', 
    #'Religion', 
    #'Indexical Error: Other', 
    #'Indexical Error: Time', 
    'Science', 
    #'Advertising', 
    #'Paranormal'
]

class TruthfulQA:
    def __init__(self, split='validation'):
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split=split)
        self.dataset = [d for d in dataset if d['type'] == 'Non-Adversarial' \
            and d['category'] in ALLOWED_CATEGORIES \
            and len(d['incorrect_answers']) >= 3
        ]
        logger.info(f"Loaded TruthfulQA dataset ({split} split, {len(self.dataset)} examples)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        correct_answer = d['best_answer']
        d['choices'] = d['incorrect_answers'][:3] + [correct_answer]
        random.shuffle(d['choices'])
        d['answer'] = d['choices'].index(correct_answer)
        d['correct_answer'] = CHOICE_LETTERS[d['answer']]
        d.pop('best_answer')
        d.pop('correct_answers')

        incorrect_answers = [c for i, c in enumerate(CHOICE_LETTERS) if i != d['answer']]
        d['incorrect_answers'] = incorrect_answers

        choices_string = '\n'.join([str(x+'. '+y) for x, y in zip(CHOICE_LETTERS, d['choices'])])
        d['choices_string'] = choices_string
        return d

if __name__ == '__main__':
    dataset = TruthfulQA()
    pdb.set_trace()