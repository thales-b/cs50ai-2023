import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    dir_content = os.listdir(directory)

    for filename in dir_content:
        if filename.endswith('.txt'):
            path = os.path.join(directory, filename)
            with open(path, 'r') as f:
                content = f.read()
                files[filename] = content

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized_doc = [
        word for word in nltk.word_tokenize(document.lower())
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english")
    ]

    return tokenized_doc


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    words = set(word for document in documents.values() for word in document)

    for word in words:
        count = sum(1 for document in documents.values() if word in document)
        idfs[word] = math.log(len(documents) / count)

    return idfs


def tf(word, words):
    count = words.count(word)
    return count


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    scores = {}
    for filename, words in files.items():
        score = 0
        for word in query:
            tf_idf = tf(word, words) * idfs.get(word, 0)
            score += tf_idf
        scores[filename] = score

    top_files = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n]

    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = []

    for sentence, words in sentences.items():
        matching_word_measure = sum(idfs.get(word, 0) for word in query if word in words)
        query_term_density = sum(1 for word in words if word in query) / len(words) if len(words) > 0 else 0

        sentence_scores.append((sentence, matching_word_measure, query_term_density))

    sentence_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_sentences = [sentence for sentence, _, _ in sentence_scores[:n]]

    return top_sentences


if __name__ == "__main__":
    main()
