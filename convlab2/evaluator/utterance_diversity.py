from collections import Counter
import math
from statistics import mean
import random
import json
from lexical_diversity import lex_div as ld


def get_diversity_metrics(user_responses):
    """
    Get lexical diversity metrics for all user utterances from all dialogs.
    inspired by https://github.com/Tomiinek/MultiWOZ_Evaluation
    and https://github.com/kristopherkyle/lexical_diversity/

    in: user_responses, list
    out: richness metrics, dict
    """
    avg_lengths, total_utterances = 0, 0
    unique_grams = [Counter() for _ in range(3)]
    all_tokens = []

    for utterance in user_responses:
        # can also use ld.flemmatize which supersets ld.tokenize functionality
        tokens = ld.tokenize(utterance)
        all_tokens.extend(tokens)

        avg_lengths += len(tokens)
        total_utterances += 1

        unique_grams[0].update(tokens)
        unique_grams[1].update(
            [(a, b) for a, b in zip(tokens, tokens[1:])])
        unique_grams[2].update(
            [(a, b, c) for a, b, c in zip(
                tokens, tokens[1:], tokens[2:])])

    # 1. Number of unique uni/big/tri-grams
    # unigram count -- number of unique tokens/words among
    # all utterances
    unique_grams_count = [len(c) for c in unique_grams]

    # 2. Average utterance length
    try:
        avg_utterance_length = avg_lengths / total_utterances
    except Exception:
        avg_utterance_length = 0

    # 3. Entropy, conditional entropy
    total = sum(v for v in unique_grams[0].values())
    probs = [(u/total) for u in unique_grams[0].values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)

    cond = [unique_grams[1][
        (h, w)]/unique_grams[0][h] for h, w in unique_grams[1]]
    join = [unique_grams[1][
        (h, w)]/total for h, w in unique_grams[1]]
    cond_entropy_bigram = -sum(
        j * math.log(c, 2) for c, j in zip(cond, join))

    # 4. Lexical diversity metrics from `lexical_diversity` library
    # ttr: ratio of unique to all tokens
    # ttr = ld.ttr(all_tokens)
    # also see: root_ttr, log_ttr, maas_ttr

    # mean segmental TTR, also see mattr
    msttr = ld.msttr(all_tokens, window_length=50)

    # Hypergeometric distribution D
    # A more straightforward and reliable implementation of vocD
    # (Malvern, Richards, Chipere, & Duran, 2004)
    # as per McCarthy and Jarvis (2007, 2010).
    hdd = ld.hdd(all_tokens)

    # Measure of lexical textual diversity (MTLD)
    # based on McCarthy and Jarvis (2010).
    mtld = ld.mtld(all_tokens)
    # mtld_ma_bid = ld.mtld_ma_bid(all_tokens, min=10)
    # mtld_ma_wrap = ld.mtld_ma_wrap(all_tokens, min=10)

    return {
        'total_utterances': total_utterances,
        'total_tokens': len(all_tokens),
        'num_unigrams': unique_grams_count[0],
        'num_bigrams': unique_grams_count[1],
        'num_trigrams': unique_grams_count[2],
        'avg_utterance_length': avg_utterance_length,
        'entropy': entropy,
        'cond_entropy_bigram': cond_entropy_bigram,
        'msttr': msttr,
        'hdd': hdd,
        'mtld': mtld,
    }


def get_golder_utterances_from_dataset(
        dataset, data_key):
    golden_utts = []

    for key, sess in dataset.items():
        for i, turn in enumerate(sess['log']):
            if data_key == 'usr' and i % 2 == 0:
                golden_utts.append(turn['text'])
            if data_key == 'sys' and i % 2 == 1:
                golden_utts.append(turn['text'])

    return golden_utts


def get_diversity_from_dataset(dataset_path, data_key='usr', sample=None):
    """
    calculate diversity metrics for selected dataset
    dataset_path - str, path to .json
    data_key - 'usr' or 'sys'
    sample - int, calculate metrics on 'sample' conversations
    """
    dataset = json.load(open(dataset_path, 'r'))
    print('read in ', len(dataset), 'conversations')
    diversity = {}

    if sample:
        diversity_list = []
        # repeat sampling 100 times
        for i in range(1000):
            sampled_dataset = dict(random.sample(dataset.items(), sample))
            if i % 20 == 0:
                print('iteration:', i, ', sampled ', sample, 'conversations')
            golden_utts = get_golder_utterances_from_dataset(
                sampled_dataset, data_key)
            diversity_list.append(get_diversity_metrics(golden_utts))

        keys = diversity_list[0].keys()
        for key in keys:
            diversity[key] = mean([d[key] for d in diversity_list])

    else:
        golden_utts = get_golder_utterances_from_dataset(
            dataset, data_key)
        diversity = get_diversity_metrics(golden_utts)

    for diversity_metric, diversity_result in diversity.items():
        print('{}: {}'.format(
            diversity_metric, format(diversity_result, '.2f')))
    return diversity
