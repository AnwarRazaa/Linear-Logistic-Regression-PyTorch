import numpy as np
import pandas as pd
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report


def load_and_preprocess(train_path, test_path, labels_path):
    # load all csv files
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_labels = pd.read_csv(labels_path)

    # merge test with labels since test.csv doesnt have Survived column
    df_test = df_test.merge(df_labels[['PassengerId', 'Survived']], on='PassengerId')

    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
    print(f"Train columns: {list(df_train.columns)}")

    # drop columns that wont help the model
    # PassengerId = just an ID number, not useful
    # Name = text, hard to use without NLP
    # Ticket = random codes, no pattern
    # Cabin = 77% missing values, not reliable
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_train = df_train.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols)

    # handle missing values using median and mode
    for df in [df_train, df_test]:
        df['Age'] = df['Age'].fillna(df_train['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df_train['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df_train['Fare'].median())

    # convert Sex to numbers: male=0, female=1
    # models cant understand text, they need numbers
    for df in [df_train, df_test]:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # one-hot encode embarked (C, Q, S become separate columns)
    df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked', dtype=int)
    df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked', dtype=int)

    print(f"\nAfter preprocessing:")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"Columns: {list(df_train.columns)}")
    print(f"\nSurvival distribution in train:")
    print(df_train['Survived'].value_counts())

    return df_train, df_test

def separate_features_target(df):
    # split dataframe into features (X) and target (Y)
    X = df.drop(columns=['Survived']).values
    Y = df['Survived'].values.reshape(-1, 1)
    return X, Y


def split_data(X, Y, val_ratio=0.15):
    # 85/15 split for train/val
    n = len(X)
    split = int(n * (1 - val_ratio))
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def compute_mean(X):
    return np.mean(X, axis=0)


def compute_std(X):
    return np.std(X, axis=0)


def normalize(X, mean, std):
    # z-score normalization with clipping
    std_safe = np.where(std == 0, 1, std)
    X_norm = (X - mean) / std_safe
    return np.clip(X_norm, -5, 5)

class TitanicDataset(Dataset):
    # wraps data so DataLoader can create batches
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        # returns total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # returns one sample at given index
        return self.X[idx], self.Y[idx]


class LogisticRegressionNetwork:
    def __init__(self, n_features, init_type='random'):
        # single layer: 9 inputs -> 1 output
        if init_type == 'random':
            self.W = np.random.randn(n_features, 1) * 0.01
        elif init_type == 'zeros':
            self.W = np.zeros((n_features, 1))
        elif init_type == 'he':
            self.W = np.random.randn(n_features, 1) * np.sqrt(2.0 / n_features)

        self.b = np.zeros((1, 1))

    def sigmoid(self, z):
        # clips z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def feed_forward(self, X):
        # z = weighted sum of inputs
        # a = sigmoid(z) = predicted probability of survival
        z = X @ self.W + self.b
        self.a = self.sigmoid(z)
        return self.a

    def compute_loss(self, Y):
        # binary cross entropy - measures how wrong our predictions are
        # if actual=1 and we predict 0.9 -> small loss (good)
        # if actual=1 and we predict 0.1 -> big loss (bad)
        m = len(Y)
        eps = 1e-8
        loss = -(1/m) * np.sum(Y * np.log(self.a + eps) + (1 - Y) * np.log(1 - self.a + eps))
        return loss

    def compute_gradients(self, X, Y):
        # how much each weight contributed to the error
        # error = prediction - actual (simple because of sigmoid + cross entropy)
        m = len(Y)
        error = self.a - Y
        self.dW = (1/m) * (X.T @ error)
        self.db = (1/m) * np.sum(error, axis=0, keepdims=True)

    def update_weights(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

def train(model, train_loader, val_X, val_Y, epochs, lr):
    # store metrics for each epoch to plot later
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for X_batch, Y_batch in train_loader:
            # convert pytorch tensors to numpy arrays
            X_batch = X_batch.numpy()
            Y_batch = Y_batch.numpy()

            # forward pass then update weights
            model.feed_forward(X_batch)
            epoch_loss += model.compute_loss(Y_batch) * len(Y_batch)
            model.compute_gradients(X_batch, Y_batch)
            model.update_weights(lr)

            # if prediction >= 0.5 -> survived (1), else died (0)
            preds = (model.a >= 0.5).astype(int)
            correct += np.sum(preds == Y_batch)
            total += len(Y_batch)

        # calculate average loss and accuracy for this epoch
        train_loss = epoch_loss / total
        train_acc = correct / total

        # check how model does on validation data
        model.feed_forward(val_X)
        val_loss = model.compute_loss(val_Y)
        val_preds = (model.a >= 0.5).astype(int)
        val_acc = np.mean(val_preds == val_Y)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # print every 10 epochs to track progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def test(model, X_test, Y_test):
    # run model on test data and get predictions
    model.feed_forward(X_test)
    test_loss = model.compute_loss(Y_test)
    preds = (model.a >= 0.5).astype(int)
    test_acc = np.mean(preds == Y_test)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(Y_test, preds, target_names=['Died', 'Survived']))

    # shows how many correct/wrong predictions for each class
    cm = confusion_matrix(Y_test, preds)
    print(f"Confusion Matrix:\n{cm}")

    return test_loss, test_acc, preds, cm


def save_model(model, mean, std, path):
    # save weights and normalization stats so we can load later
    data = {'W': model.W, 'b': model.b, 'mean': mean, 'std': std}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Model saved to {path}")


def plot_results(train_losses, val_losses, train_accs, val_accs, cm, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # plot 1: loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # plot 2: accuracy curve
    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()

    # plot 3: confusion matrix as heatmap
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks([0, 1], ['Died', 'Survived'])
    plt.yticks([0, 1], ['Died', 'Survived'])
    # put numbers inside each cell
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=14)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_experiments', action='store_true')
    args = parser.parse_args()

    train_path = "./Task-2 Dataset (Titanic)/train.csv"
    test_path = "./Task-2 Dataset (Titanic)/test.csv"
    labels_path = "./Task-2 Dataset (Titanic)/gender_submission.csv"

    df_train, df_test = load_and_preprocess(train_path, test_path, labels_path)

    X_full, Y_full = separate_features_target(df_train)
    X_test, Y_test = separate_features_target(df_test)
    X_train, Y_train, X_val, Y_val = split_data(X_full, Y_full)

    # normalize using training stats
    mean = compute_mean(X_train)
    std = compute_std(X_train)
    X_train = normalize(X_train, mean, std)
    X_val = normalize(X_val, mean, std)
    X_test = normalize(X_test, mean, std)

    # create batches of 32 samples
    train_dataset = TitanicDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    if not args.run_experiments:
        # single run
        lr = 0.1
        epochs = 100
        model = LogisticRegressionNetwork(n_features=9, init_type='he')

        print("\n--- Training ---")
        train_losses, val_losses, train_accs, val_accs = train(model, train_loader, X_val, Y_val, epochs, lr)

        test_loss, test_acc, test_preds, cm = test(model, X_test, Y_test)

        os.makedirs("saved_models", exist_ok=True)
        save_model(model, mean, std, "saved_models/logistic_model.pkl")
        plot_results(train_losses, val_losses, train_accs, val_accs, cm, "results/task2/single_run")

    else:
        # run multiple experiments with different configs
        experiments = [
            {'lr': 0.1,    'epochs': 100, 'init': 'he'},
            {'lr': 0.01,   'epochs': 100, 'init': 'he'},
            {'lr': 0.5,    'epochs': 100, 'init': 'he'},
            {'lr': 0.001,  'epochs': 100, 'init': 'he'},
            {'lr': 0.1,    'epochs': 200, 'init': 'he'},
            {'lr': 0.1,    'epochs': 50,  'init': 'he'},
            {'lr': 0.1,    'epochs': 100, 'init': 'random'},
            {'lr': 0.1,    'epochs': 100, 'init': 'zeros'},
            {'lr': 0.01,   'epochs': 200, 'init': 'random'},
            {'lr': 0.5,    'epochs': 100, 'init': 'zeros'},
        ]

        for i, exp in enumerate(experiments):
            print(f"\n{'='*50}")
            print(f"Experiment {i+1}/10 - lr={exp['lr']}, epochs={exp['epochs']}, init={exp['init']}")
            print(f"{'='*50}")

            model = LogisticRegressionNetwork(n_features=9, init_type=exp['init'])
            train_losses, val_losses, train_accs, val_accs = train(
                model, train_loader, X_val, Y_val, exp['epochs'], exp['lr']
            )
            test_loss, test_acc, test_preds, cm = test(model, X_test, Y_test)

            # save each experiment in its own folder
            folder = f"results/task2/exp{i+1}_lr{exp['lr']}_ep{exp['epochs']}_{exp['init']}"
            plot_results(train_losses, val_losses, train_accs, val_accs, cm, folder)

        print("\nAll experiments done!")
