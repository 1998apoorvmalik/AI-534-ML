import argparse
from collections import Counter

parser = argparse.ArgumentParser(description='Remove single occurrence words from training file.')
parser.add_argument('-i', '--input', default='train.txt', type=str, help='Path to the training file')
parser.add_argument('-o', '--output', default='filter_train.txt', type=str, help='Path to the filtered training file')
parser.add_argument('-c', '--count', default=1, type=int, help='Minimum number of occurrences for a word to be kept')

args = parser.parse_args()

# Step 1: Read the file and tokenize the text
with open(args.input, 'r') as file:
    lines = file.readlines()

words = []
cleaned_lines = []

# Adding words to the list but skip the labels
for line in lines:
    label, sentence = line.split('\t')
    tokens = sentence.split()
    words.extend(tokens)  # Extend the list with tokens from the sentence
    cleaned_lines.append(label)  # Start the cleaned_lines with just the label

# Step 2: Count the occurrences of each word
word_counts = Counter(words)

# Step 3: Remove words with single count from sentences and count them
removed_words_count = 0
for i, line in enumerate(lines):
    label, sentence = line.split('\t')
    tokens = sentence.split()
    # Clean tokens that are not labels and their count is more than 1
    cleaned_tokens = [token for token in tokens if word_counts[token] > args.count]
    removed_words_count += len(tokens) - len(cleaned_tokens)
    # Reconstruct the cleaned line with the label
    if len(cleaned_tokens) > 0:
        cleaned_lines[i] = label + '\t' + ' '.join(cleaned_tokens)
    else:
        cleaned_lines[i] = label + '\t' + ' '.join(tokens)

# Step 4: Write the cleaned sentences to a new file
with open(args.output, 'w') as file:
    for line in cleaned_lines:
        if len(line.strip()) == 1:
            print(line)
            continue
        file.write(line + '\n')

# Display the number of removed words
print(f"Number of words removed: {removed_words_count}")
