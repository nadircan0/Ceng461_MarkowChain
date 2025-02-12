from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random

# ------------------------------
# STEP 1: Load Dataset and Compute Probabilities
# ------------------------------

# Read dataset and add '*' as end-of-word symbol
def load_words(file_path):
    with open(file_path, "r") as file:
        words = file.read().splitlines()
    words = [word + '*' for word in words]
    return words


# Compute P(L0) - Initial probabilities
def compute_p_l0(words):
    start_counts = Counter(word[0] for word in words)
    total_starts = sum(start_counts.values())
    p_l0 = {char: count / total_starts for char, count in start_counts.items()}
    p_l0['*'] = 0.0  
    return p_l0


# Compute P(LN | LN-1) - Transition probabilities
def compute_p_ln_given_ln_minus_1(words):
    transition_counts = Counter()
    transition_totals = Counter()

    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            transition_counts[pair] += 1
            transition_totals[word[i]] += 1


    transition_probs = defaultdict(dict)
    for (prev, next_), count in transition_counts.items():
        transition_probs[prev][next_] = count / transition_totals[prev]


    transition_probs['*']['*'] = 1.0  
    for char in transition_probs:
        if '*' not in transition_probs[char]:
            transition_probs[char]['*'] = 0.05  

    return transition_probs


# Print probabilities in the desired format
def print_probabilities(initial_prob, transition_prob):
    # Initial probabilities
    print("\nInitial Probabilities:")
    print("    P(L0)")
    for char in sorted(initial_prob.keys()):
        print(f"{char}  {initial_prob[char]:.4f}")

    # Transition probabilities
    print("\nTransition Probabilities:")
    all_chars = sorted(set(initial_prob.keys()))
    print(f"{' ':<4} |", end="")
    for char in all_chars:
        print(f"{char:<8}", end="")
    print("\n" + "-" * 220)

    for char1 in all_chars:
        print(f"{char1:<4} |", end="")
        for char2 in all_chars:
            prob = transition_prob[char1].get(char2, 0)
            print(f"{prob:<8.5f}", end="")
        print()


# ------------------------------
# STEP 2: Calculate Average Word Length
# ------------------------------

def calculate_average_word_length(words):
    total_length = sum(len(word) - 1 for word in words)
    return total_length / len(words)


# ------------------------------
# STEP 3: Compute P(LN) for N=1 to 5
# ------------------------------

def calc_prior_prob1(words, N):
    nth_counts = Counter()
    total_counts = 0
    for word in words:
        if len(word) > N:
            nth_counts[word[N]] += 1
            total_counts += 1
    return {char: count / total_counts for char, count in nth_counts.items()}


def plot_prior_prob1(words):
    for N in range(1, 6):
        p_ln = calc_prior_prob1(words, N)
        labels, values = zip(*sorted(p_ln.items()))
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title(f"P(L{N}) Distribution")
        plt.xlabel("Letters")
        plt.ylabel("Probability")
        plt.show()

# ------------------------------
# STEP 4: Predict P(LN) using prior probabilities
# ------------------------------

def calc_prior_prob2(p_l0, p_ln_given_ln_minus_1, N):
    p_ln = p_l0.copy()
    for _ in range(N - 1):
        new_p_ln = defaultdict(float)
        for prev, prob_prev in p_ln.items():
            if prev in p_ln_given_ln_minus_1:
                for next_, prob_transition in p_ln_given_ln_minus_1[prev].items():
                    new_p_ln[next_] += prob_prev * prob_transition
        p_ln = new_p_ln
    return p_ln


def plot_prior_prob2(p_l0, p_ln_given_ln_minus_1):
    for N in range(1, 6):
        p_ln = calc_prior_prob2(p_l0, p_ln_given_ln_minus_1, N)
        labels, values = zip(*sorted(p_ln.items()))
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title(f"P(L{N}) Distribution (Using PriorProb2)")
        plt.xlabel("Letters")
        plt.ylabel("Probability")
        plt.show()

# ------------------------------
# STEP 5: Calculate Word Probabilities
# ------------------------------

def calc_word_prob(p_l0, p_ln_given_ln_minus_1, word):
    prob = p_l0.get(word[0], 0)
    for i in range(len(word) - 1):
        prev = word[i]
        next_ = word[i + 1]
        prob *= p_ln_given_ln_minus_1.get(prev, {}).get(next_, 0)
    return prob


# ------------------------------
# STEP 6: Generate Random Words
# ------------------------------

def generate_words(p_l0, p_ln_given_ln_minus_1, M):
    words = []
    for _ in range(M):
        word = []
        curr = random.choices(list(p_l0.keys()), weights=p_l0.values())[0]
        word.append(curr)

        while curr != '*':
            next_chars = list(p_ln_given_ln_minus_1[curr].keys())
            next_probs = list(p_ln_given_ln_minus_1[curr].values())
            curr = random.choices(next_chars, weights=next_probs)[0]
            word.append(curr)
        words.append("".join(word))
    return words


# ------------------------------
# STEP 7: Generate Synthetic Dataset and Calculate Average Length
# ------------------------------

def estimate_average_length(p_l0, p_ln_given_ln_minus_1, size):
    synthetic_words = generate_words(p_l0, p_ln_given_ln_minus_1, size)
    total_length = sum(len(word) - 1 for word in synthetic_words)
    return total_length / size

# ------------------------------
# STEP 8: Higher-Order Markov Chains (Bonus Part)
# ------------------------------

def compute_higher_order_transitions(words, k):
    """Calculate k-th order Markov Chain transition probabilities."""
    transition_counts = Counter()
    transition_totals = Counter()

    for word in words:
        padded_word = "#" * k + word  # Add padding to handle k-length history
        for i in range(len(word)):
            k_tuple = tuple(padded_word[i:i + k])  # Create k-length sequence
            next_char = padded_word[i + k]  # Next character after k-tuple
            transition_counts[(k_tuple, next_char)] += 1
            transition_totals[k_tuple] += 1

    # Calculate probabilities
    higher_order_probs = defaultdict(dict)
    for (k_tuple, next_char), count in transition_counts.items():
        higher_order_probs[k_tuple][next_char] = count / transition_totals[k_tuple]

    return higher_order_probs


def generate_words_higher_order(probs, k, M):
    """Generate M random words using k-th order Markov Chain."""
    words = []

    for _ in range(M):
        word = []
        # Start with a random k-tuple
        curr_k_tuple = random.choice(list(probs.keys()))
        word.extend(curr_k_tuple)

        while True:
            if curr_k_tuple not in probs:
                break

            next_chars = list(probs[curr_k_tuple].keys())
            next_probs = list(probs[curr_k_tuple].values())
            next_char = random.choices(next_chars, weights=next_probs)[0]

            if next_char == '*':  # Stop when reaching end-of-word symbol
                break

            word.append(next_char)
            curr_k_tuple = tuple(word[-k:])  # Update k-tuple

        words.append(''.join(word) + '*')

    return words




# ------------------------------
# MAIN SCRIPT
# ------------------------------

file_path = 'corncob_lowercase.txt'
words = load_words(file_path)

# Task 1
p_l0 = compute_p_l0(words)
p_ln_given_ln_minus_1 = compute_p_ln_given_ln_minus_1(words)
print_probabilities(p_l0, p_ln_given_ln_minus_1)

# Task 2
avg_length = calculate_average_word_length(words)
print(f"\nAverage Word Length: {avg_length:.2f}")

# Task 3
plot_prior_prob1(words)

# Task 4
plot_prior_prob2(p_l0, p_ln_given_ln_minus_1)

# Task 5
test_words = ["sad*", "exchange*", "antidisestablishmentarianism*", "qwerty*", "zzzz*", "ae*"]
print("\nWord Probabilities:")
for word in test_words:
    prob = calc_word_prob(p_l0, p_ln_given_ln_minus_1, word)
    print(f"Word: {word.ljust(40)} Probability: {prob:.10f}")

# Task 6
generated_words = generate_words(p_l0, p_ln_given_ln_minus_1, 10)
print("\nGenerated Words:")
for word in generated_words:
    print(word)

# Task 7
synthetic_avg_length = estimate_average_length(p_l0, p_ln_given_ln_minus_1, 100000)
print("\nSynthetic Average Word Length (100,000 Words):")
print(f"{synthetic_avg_length:.2f}")

# Task 8 - Higher-Order Markov Chains (Bonus Part)
print("\nTask 8: Higher-Order Markov Chains (k=2)")
k = 2 # Set the order of Markov Chain
higher_order_probs = compute_higher_order_transitions(words, k)

# Generate 10 words using higher-order Markov Chain
generated_words = generate_words_higher_order(higher_order_probs, k, 10)
print("\nGenerated Words (Higher-Order Markov Chain):")
for word in generated_words:
    print(word)

# Synthetic Average Word Length (Higher-Order)
synthetic_words = generate_words_higher_order(higher_order_probs, k, 100000)
synthetic_avg_length = sum(len(word) - 1 for word in synthetic_words) / len(synthetic_words)
print("\nSynthetic Average Word Length (Higher-Order):")
print(f"{synthetic_avg_length:.2f}")
