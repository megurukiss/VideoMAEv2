import torch

label_map={'restrainer_interaction': 0, 'unsupported_rearing': 1,
           'running': 2, 'immobility': 3, 'idle_actions': 4, 
           'interaction_with_partner': 5, 'climbing_on_side': 6}

def ground_truth_decoder(labels,num_classes=len(label_map)):
    """
        Decode the ground truth labels into a tensor.
        Args:
        - labels (list): list of strings representing the labels. examples: ['restrainer_interaction&running', 'immobility&idle_actions']
        - num_classes (int): number of classes in the dataset.
        
        Returns:
        - decoded (torch.Tensor): tensor of shape (len(labels), num_classes) with 1s in the corresponding positions.
        examples: [[0.5,0.5,0,0,0,0,0],[0,0,0,0,0,0,1]]
    """
    
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
    sigmoids = torch.softmax(outputs, dim=1)
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
    - overall accuracy (float): percentage of completely correct samples.
    """
    
    preds=custom_multi_label_pred(outputs,threshold)
    # preds_non_empty=(preds!=0)
    # targets_non_empty=(targets!=0)
    # # get indexes where both are non empty
    # non_empty=preds_non_empty*targets_non_empty
    
    # # correct = (preds == targets).float()
    
    # # for non empty indexes, get the minimum value
    # vals=torch.stack([preds[non_empty], targets[non_empty]]).min(dim=0).values
    
    # # count correct values
    # count_correct = vals.sum()
    count_correct=torch.sum(torch.min(preds, targets))
    count_total = targets.sum()
    overall_acc=count_correct/count_total
    return overall_acc

def multi_label_seperate_accuracy(outputs,targets,threshold=0.7):
    """
    Calculate accuracy for multi-label classification.
    Args:
    - outputs (torch.Tensor): raw logits from the model (before sigmoid).
    - targets (torch.Tensor): ground truth binary labels.
    - threshold (float): threshold for converting probabilities to binary output.
    
    Returns:
    -  overall accuracy (float): percentage of completely correct samples.
    -  preds (torch.Tensor): predicted labels as a binary tensor.
    """
    
    preds=custom_multi_label_pred(outputs,threshold)
    
    # preds_non_empty=(preds!=0)
    # targets_non_empty=(targets!=0)
    # # get indexes where both are non empty
    # non_empty=preds_non_empty*targets_non_empty
    
    # # correct = (preds == targets).float()
    
    # # for non empty indexes, get the minimum value
    # vals=torch.stack([preds[non_empty], targets[non_empty]]).min(dim=0).values
    # count_correct = vals.sum()
    count_correct=torch.sum(torch.min(preds, targets))
    count_total = targets.sum()
    overall_acc=count_correct/count_total
    # acc_per_label = correct.sum(dim=0) / targets.sum(dim=0)
    # # if there are no samples for a label, set accuracy to 0
    return overall_acc,torch.softmax(outputs, dim=1)


def multi_label_confusion_matrix(ground_truth_labels,pred_labels):
    """
    Calculate confusion matrix for multi-label classification.
    Args:
    - ground_truth_labels (torch.Tensor): ground truth binary labels. torch.tensor, [[0.5,0.5,0],[0,0,1],[0.5,0,0.5]], if 2 labels overlap, it is considered as 1.
    - pred_labels (torch.Tensor): predicted binary labels. same shape as ground_truth_labels.
    """
    
    # emopty confusion matrix
    length=len(ground_truth_labels[0])
    confusion_matrix=torch.zeros((length,length))
    
    
    for i in range(len(ground_truth_labels)):
        for j in range(length):
            # count TP
            if ground_truth_labels[i][j]!=0 and pred_labels[i][j]!=0:
                confusion_matrix[j][j]+=1
            if ground_truth_labels[i][j]==pred_labels[i][j] and ground_truth_labels[i][j]==1:
                confusion_matrix[j][j]+=1
            
            # count FP
            if ground_truth_labels[i][j]==0 and pred_labels[i][j]==1:
                pass
    
    