"""
NLP Utility Functions Collection
Contains data preprocessing, prediction, and evaluation metrics calculation functions
"""
from flair.data import Sentence

def prepare_dataset(dataset):
    """
    Prepares a dataset for NLP processing by converting each row into a Sentence object with an associated label.
    
    This function iterates through a given dataset, extracting the text from the 'text' field to create Sentence objects.
    It converts the 'label' field to a string to ensure label compatibility, adds this label to the Sentence,
    and collects all sentences in a list for return.
    
    Args:
        dataset (iterable): A collection of data points, each with a 'text' and 'label' field.
        
    Returns:
        list: A list of Sentence objects, each labeled with the string-converted 'label'.
    """
    sentences = []
    for row in dataset:
        # Extract text and create a Sentence object
        text = row['text']
        sentence = Sentence(text)
        
        # Convert label to string and add as a label to the sentence
        sentence.add_label('label', str(row['label']))
        
        # Append the processed sentence to the list
        sentences.append(sentence)
    
    return sentences

def get_prediction_v2(text):
    """
    Generate prediction for input text using a pre-trained transformer model.
    
    Args:
        text (str): Input text to classify
        
    Returns:
        tuple: A tuple containing:
            - probs (torch.Tensor): Softmax probabilities for all classes
            - max_prob_index (int): Index of the predicted class
            - predicted_label (str): Human-readable label of the predicted class
            
    Note:
        Requires 'tokenizer', 'model', 'device', and 'id2label' to be initialized
        in the global scope.
    """
    # Tokenize input text with padding and truncation for consistent input size
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=250, 
        return_tensors="pt"
    ).to(device)
    
    # Forward pass through the model to get raw logits
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
    
    # Convert logits to probabilities using softmax activation
    probs = outputs.logits.softmax(dim=1)
    
    # Find the class with highest probability
    max_prob_index = probs.argmax(dim=1).item()
    
    # Map predicted index to human-readable label
    predicted_label = id2label[max_prob_index]
    
    return probs, max_prob_index, predicted_label

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """
    Compute evaluation metrics for model predictions.
    
    This function calculates accuracy, precision, recall, and F1-score
    using macro averaging, which treats all classes equally regardless
    of their frequency in the dataset.
    
    Args:
        pred: Prediction object containing:
            - label_ids: True labels as integers
            - predictions: Model output logits/probabilities
            
    Returns:
        dict: Dictionary containing computed metrics:
            - 'Accuracy': Overall classification accuracy
            - 'F1': Macro-averaged F1-score
            - 'Precision': Macro-averaged precision
            - 'Recall': Macro-averaged recall
    """
    # Extract true labels from prediction object
    labels = pred.label_ids
    
    # Convert logits to predicted class indices by taking argmax
    preds = pred.predictions.argmax(-1)
    
    # Calculate precision, recall, and F1-score with macro averaging
    # The underscore catches the 'support' value which we don't need
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    
    # Calculate overall accuracy
    acc = accuracy_score(labels, preds)
    
    # Return metrics as dictionary for HuggingFace Trainer
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

