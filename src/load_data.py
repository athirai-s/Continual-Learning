from datasets import load_dataset


def load_tgqa(config="TGQA_Story_TG_Trans"):
    """
    Load TGQA dataset from HuggingFace.
    """
    dataset = load_dataset("sxiong/TGQA", config)
    return dataset


def load_tsqa():
    """
    Load Time-Sensitive QA dataset.
    """
    dataset = load_dataset("Catkamakura/ts-qa")
    return dataset

