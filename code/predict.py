from argparse import ArgumentParser
from tensorflow.keras.models import *
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

from my_utils import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    #I am a little confused about what resources_path will be, so I did a little handling for two cases here
    model_name = "model.hdf5"
    if not resources_path.endswith(".hdf5"):
        model_name = resources_path+"/"+model_name
    else:
        model_name = resources_path
        resources_path = resources_path.split("/")
        resources_path = "".join(resources_path[0:len(resources_path)-1])
    print("Loading model...")
    model = load_model(model_name)

    #load the saved vocabularies
    print("Loading saved vocabularies...")
    with open(resources_path+"/"+"x_vocab_uni_only.utf8", 'r', encoding="utf-8-sig") as file:
        vocab = file.read()
    vocab = json.loads(vocab)

    with open(resources_path+"/"+"y_vocab.utf8", 'r', encoding="utf-8-sig") as file:
        labels_vocab = file.read()
    labels_vocab = json.loads(labels_vocab)
    
    id_to_label = {v:k for k,v in labels_vocab.items()}

    #preparing the input data for prediction
    print("Preparing the input data for prediction...")
    sentences = load_dataset(input_path)
    X_ = make_X(sentences, vocab)
    #X_ = pad_sequences(X, truncating="pre", padding="post", maxlen=50)

    #predicting and writing to file
    print("Predicting (line by line) and writing to file... This may take a little while...")
    k = 0
    x_len = X_.shape
    with open(output_path, "w") as file:
        for x in X_:
            if x.size != 0:
                x__ = np.expand_dims(x, axis=0)
                y_pred = model.predict(x__)
                #y_pred = np.squeeze(model.predict(x__))
                line_label = [id_to_label[k] for k in np.argmax(y_pred[0], 1)]
                file.write("".join(line_label) + "\n")
            
            k = k+1
            if k % 100 < 1:
                print ("%d/%d lines done... A few more moments and everything will be done! :)" % (k,x_len[0]));


if __name__ == '__main__':
    """
    SAMPLE USAGE:
    python code/predict.py data/msr_test_gold_input.utf8 data/msr_test_gold_predicted.txt resources
    """
    args = parse_args()
    print("Please wait while the script works...")
    predict(args.input_path, args.output_path, args.resources_path)
    print("Done!")