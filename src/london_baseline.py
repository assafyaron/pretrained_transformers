# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

def main():
    # Predict "London" for each of the 500 dev examples
    predictions = ["London"] * 500
    total_examples, num_correct = utils.evaluate_places("birth_dev.tsv", predictions)
    accuracy = num_correct / total_examples
    print(f"Accuracy: {num_correct} / {total_examples} = {accuracy:.4f}")

if __name__ == "__main__":
    main()