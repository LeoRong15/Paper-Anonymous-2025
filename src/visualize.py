import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_metrics):
    plt.figure(figsize=(14, 7))
    for i, (metric, data, color) in enumerate([
        ('Train Loss', train_losses, 'blue'),
        ('Val Accuracy', val_metrics['accuracies'], 'orange'),
        ('Val Recall', val_metrics['recalls'], 'green'),
        ('Val F1 Score', val_metrics['f1s'], 'red'),
        ('Val Precision', val_metrics['precisions'], 'purple'),
        ('Val Kappa', val_metrics['kappas'], 'brown')
    ], 1):
        plt.subplot(2, 3, i)
        plt.plot(data, label=metric, color=color)
        plt.xlabel('Epoch')
        plt.ylabel(metric.split()[-1])
        plt.title(metric)
        plt.legend()
    plt.tight_layout()
    plt.savefig('./Evaluation_Results.svg', format='svg')
    plt.close()

def save_metrics(best_metrics, filename='./training_results.txt'):
    with open(filename, 'w') as f:
        for key, value in best_metrics.items():
            f.write(f'best_{key.capitalize()}: {value:.4f}\n')
