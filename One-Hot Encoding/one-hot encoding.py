def one_hot_encoding(labels):
    vocab_list = list(set(labels))
    length = len(labels)
    output = []
    for i in range(length):
        zeros = [0] * len(vocab_list)
        pos = vocab_list.index(labels[i])
        zeros[pos] = 1
        output.append(zeros)
    
    return output, vocab_list


sample_labels = ["dog", "cat", "mouse", "dog"]
matrix, vocab = one_hot_encoding(sample_labels)

print("Encoded Matrix:")
for row in matrix:
    print(row)

print("\nDecoding Map Key:")
print(vocab)