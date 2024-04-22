import matplotlib.pyplot as plt
import numpy as np

def plot_chart_line(data, label, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='blue', label=label)
    plt.title(title)
    plt.xlabel('Date')
    # plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    return plt


def config_chart(title):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.grid(True)
    return plt


def plot_model_performance(fit_result):
    loss = fit_result.history.get('loss')
    mean_absolute_error = fit_result.history.get('mean_absolute_error')
    mean_squared_logarithmic_error = fit_result.history.get('mean_squared_logarithmic_error')

    accuracy = fit_result.history.get('accuracy')
    recall = fit_result.history.get('recall_1')
    precision = fit_result.history.get('precision_1')



    if(loss):
        plt = config_chart('Loss Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(loss)
        plt.legend(['Loss'])
        plt.show()

    if(mean_absolute_error):
        plt = config_chart('Mean Absolute Error Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(mean_absolute_error)
        plt.legend(['Mean Absolute Error'])
        plt.show()

    if(accuracy):
        plt = config_chart('Accuracy Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(accuracy, label='Training Accuracy')
        plt.show()

    if(recall):
        plt = config_chart('Recall Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(recall)
        plt.show()

    if(precision):
        plt = config_chart('Precision Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(precision)
        plt.show()
    
    # if(accuracy):
    #     plt = config_chart('Accuracy Result')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Value')
    #     plt.plot(accuracy, label='Training Accuracy')
    #     plt.show()
        
def plot_scatter_data(real_data, predicted_data):
    real_data = np.array(real_data)
    predicted_data = np.array(predicted_data)

    plt = config_chart('Scatter Actual vs. Predicted Values')
    plt.scatter(range(len(real_data)), real_data, label='Expected Result')
    plt.scatter(range(len(predicted_data)), predicted_data, label='Actual Result')
    plt.legend()
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.show()


def plot_distribution_data(real_data, predicted_data):
    real_data = np.array(real_data)
    predicted_data = np.array(predicted_data)

    plt = config_chart('Distribution of Actual and Predicted Values')
    plt.hist(real_data, bins=30, label='Actual Data', alpha=0.7)
    plt.hist(predicted_data, bins=30, label='Predictions', alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
def plot_original_vs_predicted_value(values):
    labels = ['green' if value > 0 else 'red' for value in values]
    colors = ['green' if value > 0 else 'red' for value in values]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values, color=colors, width=0.4)  # Adjust width as needed

    plt.title('Original vs. Predicted Value')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.xticks(rotation=0)  # Remove x-axis rotation
    # Gridlines
    # plt.grid(axis='y')
    plt.show()
    
def plot_direction_comparison(original_changes, predicted_changes):

    # if original_changes.shape != predicted_changes.shape:
    #   raise ValueError("Arrays must have the same shape.")

    # original_directions = np.sign(original_changes)
    # predicted_directions = np.sign(predicted_changes)

    # agreement = original_directions == predicted_directions

    # correct = np.sum(agreement)
    # incorrect = len(agreement) - correct

    binary_predictions = np.where(predicted_changes >= 0.5, 1, 0).flatten()
    correct = np.sum(original_changes == binary_predictions)
    incorrect = np.sum(original_changes != binary_predictions)

    
    mask_excluded = (predicted_changes < 0.45) | (predicted_changes > 0.55)
    binary_predictions_excluded = np.where(mask_excluded, np.where(predicted_changes >= 0.5, 1, 0), np.nan).flatten()
    correct_excluded = np.sum((original_changes == binary_predictions_excluded) & (~np.isnan(binary_predictions_excluded)))
    incorrect_excluded = np.sum((original_changes != binary_predictions_excluded) & (~np.isnan(binary_predictions_excluded)))

    labels = ['Correct Predictions', 'Incorrect Predictions']
    counts = [correct, incorrect]

    plt = config_chart('Agreement Between Original and Predicted Directions')
    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Prediction Direction')
    plt.ylabel('Count')
    plt.show()
    
    labels = ['Correct Predictions', 'Incorrect Predictions']
    counts = [correct_excluded, incorrect_excluded]

    plt = config_chart('Agreement Between Original and Predicted Directions')
    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Prediction Direction')
    plt.ylabel('Count')
    plt.show()