import torch
def compute_class_weight(labels):
    total = len(labels)
    spam_count = sum(labels)
    ham_count = total - spam_count
    weight_for_0 = total / (2 * ham_count)
    weight_for_1 = total / (2 * spam_count)
    return torch.tensor([weight_for_0, weight_for_1])