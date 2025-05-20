import numpy as np
import csv


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1
    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])
    return seqs_x

def csv_write(terms, branch, jobid, dirt, mod = 'w+'):
    csv_predicted = dirt + jobid + '_' + branch.upper() + '_predicted.csv'
    with open(csv_predicted, mod) as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        if mod == 'w+':
            if branch=='EC':
                writer.writerow(['# Predicted by PhiGnet Version 1.0.1'])
                writer.writerow(['ID', 'Branch','EC_number', 'Score', 'Function'])
            else:
                writer.writerow(['# Predicted by PhiGnet Version 1.0.1'])
                writer.writerow(['ID', 'Branch','GO_term', 'Score', 'Function'])
        else:
            for item in terms:
                sorted_rows = sorted(terms[item], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    writer.writerow([item, branch, row[0], '{:.5f}'.format(row[2]), row[1]])

    csvFile.close()


