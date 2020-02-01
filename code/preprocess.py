from argparse import ArgumentParser
from typing import Tuple

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the original file. This should contain the name of the file")
    parser.add_argument("output_path", help="The path to the folder where you want the input and label files stored, and the desired filename, without the extension")

    return parser.parse_args()


def prepare_dataset(original_file: str, input_path: str):
    """
    :param original_file; Full path to the original dataset
    :param input_path; Path to the directory to store processed files, with the filename, but no extension
    :return: None
    """
    lines = []
    labels = []
    with open(original_file, "r", encoding="utf-8-sig") as file:
        for line in file:
            line_chars, line_labels = get_chars_and_labels(line)
            lines.append(line_chars)
            labels.append(line_labels)

    with open(input_path+"_input.utf8", "w", encoding="utf-8-sig") as file:
        for line in lines:
            file.write(line+"\n")

    with open(input_path+"_labels.utf8", "w") as file:
        for label in labels:
            file.write(label+"\n")


def get_chars_and_labels(line: str) -> Tuple[str, str]:
    """
    :param line; A line from the dataset as str
    :return chars; Compressed string for the line
    :return labels; Word_segment code for the line. Same len as chars
    """
    chars = ""
    labels = ""

    words =  line.strip().split(" ")
    for word in words:
        chars += word
        
        word_len = len(word)
        if (word_len == 1):
            labels += "S"
        elif (word_len > 1):
            labels += "B"+"I"*(word_len-2)+"E"

    return chars, labels


if __name__ == '__main__':
    """
    SAMPLE USAGE:
    python code/preprocess.py data/msr_test_gold.utf8 data/msr_test_gold
    """
    args = parse_args()
    print("Please wait while the script works on the file...")
    prepare_dataset(args.input_path, args.output_path)
    print("Done!")