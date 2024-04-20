import torch

label_map={'restrainer_interaction': 0, 'unsupported_rearing': 1,
           'running': 2, 'immobility': 3, 'idle_actions': 4, 
           'interaction_with_partner': 5, 'climbing_on_side': 6}

def ground_truth_decoder(labels,num_classes=len(label_map)):
    decoded = torch.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        parts = label.split('&')
        for part in parts:
            decoded[i, label_map[part]] += 1
    return decoded/(len(parts))

def custom_multi_label_pred(outputs,threshold=0.7):
    """
    Generate predictions based on custom rules:
    - If the highest probability is greater than 0.7, only the highest is selected.
    - Otherwise, select the top two probabilities as valid labels.
    
    Args:
    - outputs (torch.Tensor): raw logits from the model.
    - threshold (float): threshold for considering only the highest label.
    
    Returns:
    - preds (torch.Tensor): predicted labels as a binary tensor.
    """
    sigmoids = torch.sigmoid(outputs)
    preds = torch.zeros_like(sigmoids)
    for i in range(sigmoids.shape[0]):  # Iterate over each sample
        top_values, top_indices = torch.topk(sigmoids[i], 2) # Get the indices of the top 2 probabilities
        
        if top_values[0] > threshold:
            preds[i, top_indices[0]] = 1.0
        else:
            preds[i, top_indices[0]] = 0.5
            preds[i, top_indices[1]] = 0.5
    
    return preds

def multi_label_accuracy(outputs, targets, threshold=0.7):
    """
    Calculate accuracy for multi-label classification.
    Args:
    - outputs (torch.Tensor): raw logits from the model (before sigmoid).
    - targets (torch.Tensor): ground truth binary labels.
    - threshold (float): threshold for converting probabilities to binary output.
    
    Returns:
    - accuracy (torch.Tensor): accuracy per label.
    - sample_accuracy (float): percentage of completely correct samples.
    """
    
    preds=custom_multi_label_pred(outputs,threshold)
    correct = (preds == targets).float()
    # add values for correct places
    vals=targets[correct==1]
    count_correct = vals.sum()
    count_total = targets.sum()
    overall_acc=count_correct/count_total
    return overall_acc
    