import matplotlib.pyplot as plt

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
    accuracy = fit_result.history.get('accuracy')
    mean_absolute_error = fit_result.history.get('mean_absolute_error')
    mean_squared_logarithmic_error = fit_result.history.get('mean_squared_logarithmic_error')

    if(loss and mean_absolute_error):
        plt = config_chart('Training Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(loss)
        plt.plot(mean_absolute_error)
        plt.legend(['Loss', 'Mean Absolute Error'])
        plt.show()


    if(accuracy):
        plt = config_chart('Accuracy Result')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(accuracy, label='Training Accuracy')
        plt.show()
    
    # if(accuracy):
    #     plt = config_chart('Accuracy Result')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Value')
    #     plt.plot(accuracy, label='Training Accuracy')
    #     plt.show()