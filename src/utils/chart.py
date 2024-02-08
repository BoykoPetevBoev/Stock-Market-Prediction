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