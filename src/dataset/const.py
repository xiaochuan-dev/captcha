# 常数

charset = '2345678abcdefgmnpwxy'
num_classes = len(charset) + 1

char2idx = {c: i + 1 for i, c in enumerate(charset)}
idx2char = {i + 1: c for i, c in enumerate(charset)}

blank = 0