from .mmlu_clinical import MMLUClinical
from .truthful_qa import TruthfulQA

DATASET_CLASS_MAP = {
    "mmlu_clinical": MMLUClinical,
    "truthful_qa": TruthfulQA
}