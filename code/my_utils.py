from typing import Tuple, List, Dict
import numpy as np

def load_dataset(input_path: str) -> List[str]:
    """
    :param input_path; Path to the input dataset
    :return sentences; List of sentences in input_file
    """
    sentences = []
    with open(input_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            sentences.append(line.strip())

    return sentences


def get_unigrams(sentence: str) -> List[str]:
    """
    :param sentence; A line from the dataset as str
    :return unigrams; List of unigrams in the line
    :return bigrams; List of bigrams in the line
    """
    unigrams = []
    for char in sentence:
        unigrams.append(char)

    return unigrams


def update_vocab(sentences: List[str], vocab: Dict[str, int]) -> Dict[str, int]:
    '''
    :param sentences; List of input sentences from the dataset
    :param vocab; Dictionary from unigram to int
    :return vocab; Updated dictionary from unigram to int
    '''

    for sentence in sentences:
        unigrams = get_unigrams(sentence)

        for k in range(len(unigrams)):
            if unigrams[k] not in vocab:
                vocab[unigrams[k]] = len(vocab)

    return vocab


def make_X(sentences: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """
    :param sentences; List of sentences
    :param unigrams_vocab; Unigram vocabulary
    :param bigrams_vocab; Bigram vocabulary
    :return X; Matrix storing all sentences' feature vector 
    """
    #print("Length of vocab:", len(vocab))
    #vocab = update_vocab(sentences, vocab)
    #print("Length of vocab:", len(vocab))
    X = []
    for sentence in sentences:
        x_temp = []
        unigrams = get_unigrams(sentence)
        for i in range(len(unigrams)):
            if unigrams[i] in vocab:
                x_temp.append(vocab[unigrams[i]])
            else:
                x_temp.append(vocab["UNK"])

        X.append(np.array(x_temp))

    X = np.array(X)
    return X