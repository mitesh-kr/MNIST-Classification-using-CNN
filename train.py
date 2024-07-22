import torch 
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Define Training model
class CNNTrainer:
    def __init__(self, model, optimizer, loss_function, epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs

    #Inititate training
    def train(self, X_train, y_train, X_validation, y_validation):
        train_loss_list = []
        train_accuracy_list = []
        validation_loss_list = []
        validation_accuracy_list = []

        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X_train), 32):
                X_batch = X_train[i:i + 32].unsqueeze(1)
                y_batch = y_train[i:i + 32]
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_function(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    #Evaluation after each epoch on train and validation testset
            self.model.eval()
            with torch.no_grad():
                train_pred = self.model(X_train.unsqueeze(1))
                train_loss = self.loss_function(train_pred, y_train)
                train_loss_list.append(train_loss.item())
                train_accuracy = torch.sum(torch.argmax(train_pred, dim=1) == y_train).item() / y_train.shape[0]
                train_accuracy_list.append(train_accuracy)

                validation_pred = self.model(X_validation.unsqueeze(1))
                validation_loss = self.loss_function(validation_pred, y_validation)
                validation_loss_list.append(validation_loss.item())
                validation_accuracy = torch.sum(torch.argmax(validation_pred, dim=1) == y_validation).item() / y_validation.shape[0]
                validation_accuracy_list.append(validation_accuracy)

            print(f'Epoch:{epoch + 1}\n Training Loss  :{train_loss:.4f}, Training Accuracy  :{train_accuracy * 100:.2f}%\n '
                  f'Validation Loss:{validation_loss:.4f}, Validation Accuracy:{validation_accuracy * 100:.2f}%')

        return train_loss_list, validation_loss_list, train_accuracy_list, validation_accuracy_list

# Plot of loss and accuarcy on traina and validation dataset
    def plot_loss_accuracy(self, epochs, metric_values_list, labels, ylabel, title, colors):
        for metric_values, label, color in zip(metric_values_list, labels, colors):
            plt.plot(range(1, epochs + 1), metric_values, label=label, color=color)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

# Evaluation on test dataset and plot of confusion matrix
    def evaluation_and_confusion_matrix(self, X_test, y_test, class_names):
        self.model.eval()
        test_pred = self.model(X_test.unsqueeze(1))
        test_loss = self.loss_function(test_pred, y_test)
        test_accuracy = torch.sum(torch.argmax(test_pred, dim=1) == y_test).item() / y_test.shape[0]
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

        test_predictions = torch.argmax(test_pred, dim=1).cpu().numpy()
        conf_matrix = confusion_matrix(y_test.cpu().numpy(), test_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
