# LSTM Chinese Word Segmenter

The task is to implement a [state-of-the-art Chinese word segmenter](https://www.aclweb.org/anthology/D18-1529/) - sequence tagging - model, encoding the output in the BEIS format with each character marked as belonging to one of the four classes - B (Beginning of a word), E (End of a word), I (Inside of a word), or S (Single character).

The model consists of bidirectional LSTMs, as descirbed in the images below:

![Bi-LSTM models](images/figure1.jpg "Bi-LSTM models") ![Imput structures to the model](images/figure2.jpg "Imput structures to the model")

[Read the report here](report.pdf)

This project was done as part of a graduate degree NLP course with Prof. Navigli (BabelNet) at Sapienza University of Rome and was graded as excellent


## How to test

Preprocess the test data:
```
python code/preprocess.py data/msr_test_gold.utf8 data/msr_test_gold
# python path_to_script path_to_input_dataset_to_preprocess path_to_store_input_processed_data
```

Make predictions:
```
python code/predict.py data/msr_test_gold_input.utf8 data/msr_test_gold_predicted.txt resources
# python path_to_script path_to_processed_input_file path_to_store_predictions
```

Score predictions:
```
python code/score.py data/msr_test_gold_predicted.txt data/msr_test_gold_labels.utf8
# python path_to_script path_to_predictions_file path_to_gold_file
```