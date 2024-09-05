from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn import metrics
import importlib
import matplotlib.pyplot as plt
import re


def cleanup_result_first(output):
    """
    This will return the first match.
    """
    output_lower = output.lower()
    keywords = {
        '**potential issue**': 'potential issue',
        'potential issue': 'potential issue',
        'red flag': 'red flag',
        '**red flag**': 'red flag'
    }
    # Find the position of each keyword in the text
    positions = {keyword: output_lower.find(keyword) for keyword in keywords}
    # Filter out keywords that are not found
    positions = {keyword: pos for keyword, pos in positions.items() if pos != -1}
    # If no keywords are found, return 'none'
    if not positions:
        return 'none'
    # Find the keyword with the smallest position
    first_keyword = min(positions, key=positions.get)
    # Return the corresponding label
    return keywords[first_keyword]



def cleanup_result_first_new(output):
    """
    This will return the first match of 'potential issue' or 'red flag'.
    """
    output_lower = output.lower()
    keywords = ['potential issue', 'red flag']
    
    # Combine the keywords into a single regex pattern
    pattern = re.compile(r'|'.join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)
    
    # Search for the first occurrence of any keyword
    match = pattern.search(output_lower)
    
    # If a match is found, return the matched keyword
    if match:
        return match.group().lower()
    
    # If no keywords are found, return 'none'
    return 'none'




def cleanup_result_last(output):
    """
    This will return the last match.
    """
    output_lower = output.lower()
    keywords = {
        '**potential issue**': 'potential issue',
        'potential issue': 'potential issue',
        'red flag': 'red flag',
        '**red flag**': 'red flag'
    }
    # Find the position of each keyword in the text
    positions = {keyword: output_lower.find(keyword) for keyword in keywords}
    # Filter out keywords that are not found
    positions = {keyword: pos for keyword, pos in positions.items() if pos != -1}
    # If no keywords are found, return 'none'
    if not positions:
        return 'none'
    # Find the keyword with the greatest position
    last_keyword = max(positions, key=positions.get)
    # Return the corresponding label
    return keywords[last_keyword]



def cleanup_result_last_new(output):
    """
    This will return the last match of 'potential issue' or 'red flag'.
    """
    output_lower = output.lower()
    keywords = ['potential issue', 'red flag']
    
    # Combine the keywords into a single regex pattern
    pattern = re.compile(r'|'.join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)
    
    # Find all occurrences of any keyword
    matches = list(pattern.finditer(output_lower))
    
    # If matches are found, return the last matched keyword
    if matches:
        return matches[-1].group().lower()
    
    # If no keywords are found, return 'none'
    return 'none'



def cleanup_result_after_result(output):
    """
    Cleans the result from the output text.
    Extracts 'red flag' or 'potential issue' that appears after 'Result:'.
    Ignores any newline characters and other surrounding text.
    """
    # Convert the output to lower case for case-insensitive matching
    output_lower = output.lower()
    
    # Define the regular expression pattern
    pattern = re.compile(r'result:\s*([\s\S]*?)(red flag|potential issue)', re.IGNORECASE)
    
    # Search for the pattern in the output
    match = pattern.search(output_lower)
    
    if match:
        # Return the matched result
        return match.group(2)
    else:
        # Return 'none' if no match is found
        return 'none'



def metrics_mine(ground_truth, predictions):
    """
    predictions is cleaned
    returns accuracy, precision, recall and F1 (macro) correct to 2dp
    """
    predictions = [prediction.lower() for prediction in predictions]
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=0) # macro calculates the precision for each class and then takes the unweighted mean - useful for an imbalanaced dataset
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average= 'macro', zero_division=0)
    return f"Accuracy: {accuracy: .2f}, Precision: {precision: .2f}, Recall: {recall: .2f}, F1: {f1: .2f}"


def metrics_mine_dict(ground_truth, predictions):
    """
    Returns a dictionary with accuracy, precision, recall, and F1 (macro) correct to 2dp.
    """
    predictions = [prediction.lower() for prediction in predictions]
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    
    metrics_dict = {
        'Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1': round(f1, 2)
    }
    return metrics_dict


def plot_confusion_matrix(ground_truth, predictions):
    """
    Plots the confusion matrix.
    """
    predictions = [prediction.lower() for prediction in predictions]
    confusion_matrix = metrics.confusion_matrix(ground_truth, predictions, labels = ['potential issue','red flag'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['potential issue','red flag'])
    cm_display.plot()
    plt.show()