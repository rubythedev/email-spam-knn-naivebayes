'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Ruby Nunez
'''

import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.
    '''

    word_freq = {}
    num_emails = 0

    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    for root, dirs, files in os.walk(email_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    email_text = f.read()
                    words = pattern.findall(email_text.lower())
                    for word in words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                    num_emails += 1
            except UnicodeDecodeError:
                continue

    return word_freq, num_emails


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    top_words = []
    counts = []

    for word, count in sorted_words[:num_features]:
        top_words.append(word)
        counts.append(count)
    
    return top_words, counts


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    '''

    num_features = len(top_words)
    feats = np.zeros((num_emails, num_features), dtype=np.int32)
    y = np.zeros(num_emails, dtype=np.int32)

    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    email_index = 0

    for root, dirs, files in os.walk(email_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    email_text = f.read()
                    words = pattern.findall(email_text.lower())
                    word_freq = {}

                    for word in words:
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1

                    for i, top_word in enumerate(top_words):
                        if top_word in word_freq:
                            feats[email_index, i] = word_freq[top_word]

                    class_dir = os.path.basename(root)
                    if class_dir == 'spam':
                        y[email_index] = 1
                    else:
                        y[email_index] = 0

                    email_index += 1
            except UnicodeDecodeError:
                continue

    return feats, y


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should be used
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should data be shuffled before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if there are originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if there are originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''

    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    num_test = int(test_prop * y.size)
    x_train = features[num_test:]
    y_train = y[num_test:]
    inds_train = inds[num_test:]
    x_test = features[:num_test]
    y_test = y[:num_test]
    inds_test = inds[:num_test]

    return x_train, y_train, inds_train, x_test, y_test, inds_test


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    
    email_texts = []
    
    for ind in inds:
        email_file = os.path.join(email_path, f"{ind}.txt")
        
        try:
            with open(email_file, 'r') as f:
                email_texts.append(f.read())
        except FileNotFoundError:
            print(f"Warning: Email file {email_file} not found.")
        except Exception as e:
            print(f"Error reading file {email_file}: {e}")
    
    return email_texts
