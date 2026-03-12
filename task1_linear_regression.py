# all imports
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def load_data(train_path, test_path):
    # Load the training and testing data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # fill missing values in training data with median of each column
    # median is better than mean because extreme values dont affect it
    for col in df_train.columns:
        median_val = df_train[col].median()
        df_train[col] = df_train[col].fillna(median_val)

    # test CSV has two problems:
    # 1. many rows have NaN (missing values) - we drop those rows
    # 2. some values are insanely large (eg HouseAge = 13 million) - we remove those rows
    df_test = df_test.dropna().reset_index(drop=True)

    # remove rows where any value is way outside the training data range
    for col in df_test.columns:
        col_min = df_train[col].min()
        col_max = df_train[col].max()
        col_range = col_max - col_min
        df_test = df_test[(df_test[col] >= col_min - col_range) & (df_test[col] <= col_max + col_range)]
    df_test = df_test.reset_index(drop=True)

    print(f"Train samples: {len(df_train)}, Test samples after cleaning: {len(df_test)}")
    return df_train, df_test

def separate_features_target(df):
    """Separate features and target from dataframe"""
    X = df.drop("target", axis=1).values       # shape: (N, 8)
    Y = df["target"].values.reshape(-1, 1)      # shape: (N, 1)
    return X, Y

def split_data(X, Y, train_ratio=0.85):
    """Split data into training and validation sets"""
    N = X.shape[0]
    split_index = int(N * train_ratio)
    
    train_X = X[:split_index]
    train_Y = Y[:split_index]
    val_X = X[split_index:]
    val_Y = Y[split_index:]
    
    return train_X, train_Y, val_X, val_Y

def compute_mean(X):
    """Compute mean of each feature (column)"""
    return np.mean(X, axis=0)    # shape: (8,)

def compute_std(X):
    """Compute standard deviation of each feature (column)"""
    return np.std(X, axis=0)     # shape: (8,)

def normalize(X, mean, std):
    # Z-score normalization: centers data around 0 with std of 1
    # also clips extreme values to [-5, 5] range
    # without clipping, outlier values in test data can break predictions
    X_norm = (X - mean) / std
    X_norm = np.clip(X_norm, -5, 5)
    return X_norm

class HousingDataset(Dataset):
    def __init__(self, X, Y):
        """
        Store the data. Think of this as a container that holds your data.
        X = features (like a table with 8 columns)
        Y = target (house prices)
        """
        self.X = X.astype(np.float32)   # convert to float32 (PyTorch needs this)
        self.Y = Y.astype(np.float32)
    
    def __getitem__(self, index):
        """
        This is called when you ask for a specific sample.
        Like saying: "Give me row number 5 from the data"
        The DataLoader calls this automatically to build batches.
        """
        return self.X[index], self.Y[index]
    
    def __len__(self):
        """
        Returns total number of samples.
        DataLoader needs this to know when to stop.
        """
        return len(self.X)

class LinearRegressionNetwork:
    def __init__(self, input_size, hidden_size, output_size=1, init_type="he"):
        # input_size = 8 (our features)
        # hidden_size = 8 or 2 (both are asked)
        # output_size = 1 (predicting one price value)
        
        # setting up weights based on which initialization method we chose
        if init_type == "he":
            # he init - scaled random values, works best in practice
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        
        elif init_type == "random":
            # plain random values from normal distribution
            self.W1 = np.random.randn(input_size, hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size)
        
        elif init_type == "zeros":
            # all zeros - included just for comparison, doesnt work well
            self.W1 = np.zeros((input_size, hidden_size))
            self.W2 = np.zeros((hidden_size, output_size))
        
        # biases start at zero, thats the standard approach
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def feed_forward(self, X):
        # Step 1: input goes to hidden layer
        # multiply features by weights and add bias
        # its just matrix multiplication: each neuron gets a weighted sum of inputs
        self.z1 = X @ self.W1 + self.b1
        
        # apply ReLU activation on hidden layer
        # ReLU just means: if value is negative, make it 0. if positive, keep it.
        # why? without activation, stacking layers is pointless (its just linear math)
        # ReLU adds non-linearity so the model can learn curved/complex patterns
        self.a1 = np.maximum(0, self.z1)
        
        # Step 2: hidden layer goes to output
        # no activation here because we want any number as price (not just positive)
        self.z2 = self.a1 @ self.W2 + self.b2
        
        return self.z2  # this is y_hat (predicted price)
    
    def compute_loss(self, y, y_hat):
        # L2 loss (mean squared error / 2)
        # measures how far off our predictions are from actual prices
        # squaring makes big errors count way more than small errors
        # dividing by 2N is just for cleaner gradient math later
        N = y.shape[0]
        loss = (1 / (2 * N)) * np.sum((y - y_hat) ** 2)
        return loss
    
    def compute_gradients(self, X, y, y_hat):
        # backpropagation - figuring out which direction to adjust each weight
        # we start from the output and work backwards (thats why its called "back" propagation)
        N = X.shape[0]
        
        # how wrong was the output layer?
        dz2 = (y_hat - y) / N
        # how much should W2 change?
        dW2 = self.a1.T @ dz2
        # how much should b2 change?
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # now go back one more layer - how wrong was the hidden layer?
        da1 = dz2 @ self.W2.T
        # ReLU derivative: if original value was > 0, gradient passes through. if <= 0, gradient = 0
        dz1 = da1 * (self.z1 > 0)
        # how much should W1 change?
        dW1 = X.T @ dz1
        # how much should b1 change?
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2, lr):
        # nudge every weight slightly in the direction that reduces the loss
        # lr (learning rate) controls how big each nudge is
        # too big = overshoots and loss explodes
        # too small = barely moves and takes forever
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


# ---- FUNCTION: r_squared ----
# measures how good our model is compared to just guessing the average price
# 1.0 = perfect predictions, 0.0 = same as guessing the mean, negative = worse than mean
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# ---- FUNCTION: train ----
# this is the main training loop where the model actually learns
# it goes through the data many times (epochs), each time adjusting weights to reduce error
def train(net, train_loader, train_X, train_Y, val_X, val_Y, n_epochs, lr):
    # lists to store loss/r2 after each epoch for plotting later
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(n_epochs):
        # go through all batches in training data
        for X_batch, Y_batch in train_loader:
            # convert pytorch tensors to numpy arrays since our model uses numpy
            X_b = X_batch.numpy()
            Y_b = Y_batch.numpy()

            # predict prices for this batch
            y_hat = net.feed_forward(X_b)

            # compute gradients (which direction to move weights)
            dW1, db1, dW2, db2 = net.compute_gradients(X_b, Y_b, y_hat)

            # update weights (nudge them to reduce error)
            net.update_weights(dW1, db1, dW2, db2, lr)

        # after all batches done, check how good the model is on full data
        train_pred = net.feed_forward(train_X)
        t_loss = net.compute_loss(train_Y, train_pred)
        t_r2 = r_squared(train_Y, train_pred)

        val_pred = net.feed_forward(val_X)
        v_loss = net.compute_loss(val_Y, val_pred)
        v_r2 = r_squared(val_Y, val_pred)

        # save for plotting
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_r2_scores.append(t_r2)
        val_r2_scores.append(v_r2)

        # print progress every 10 epochs so we can see whats happening
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Train R2: {t_r2:.4f} | Val R2: {v_r2:.4f}")

    return net, train_losses, val_losses, train_r2_scores, val_r2_scores


# ---- FUNCTION: test ----
# runs the trained model on test data (data it has never seen during training)
# this tells us how the model would perform in the real world
def test(net, test_X, test_Y):
    test_pred = net.feed_forward(test_X)
    test_loss = net.compute_loss(test_Y, test_pred)
    test_r2 = r_squared(test_Y, test_pred)

    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")

    return test_loss, test_r2, test_pred


# ---- FUNCTION: save_model ----
# saves weights + mean/std to a file so the model can be loaded and used later
# we need to save mean/std too because new data must be normalized the same way
def save_model(net, mean, std, path):
    model_data = {
        'W1': net.W1, 'b1': net.b1,
        'W2': net.W2, 'b2': net.b2,
        'mean': mean, 'std': std
    }
    pickle.dump(model_data, open(path, 'wb'))
    print(f"Model saved to {path}")


# ---- FUNCTION: plot_results ----
# generates 3 plots and saves them to the given folder
# plot 1: loss curve (how error decreased over time)
# plot 2: r2 curve (how accuracy improved over time)
# plot 3: predicted vs actual prices (visual check)
def plot_results(train_losses, val_losses, train_r2, val_r2, test_loss, test_r2, test_pred, test_Y, save_dir):
    # create the folder if it doesnt exist
    os.makedirs(save_dir, exist_ok=True)

    # plot 1: loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # plot 2: r-squared curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_r2, label='Train R2')
    plt.plot(val_r2, label='Validation R2')
    plt.axhline(y=test_r2, color='r', linestyle='--', label=f'Test R2: {test_r2:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('Training vs Validation R2 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'r2_curve.png'))
    plt.close()

    # plot 3: predicted vs actual prices for first 50 test samples
    plt.figure(figsize=(8, 8))
    plt.scatter(range(50), test_pred[:50], label='Predicted', marker='x', color='red')
    plt.scatter(range(50), test_Y[:50], label='Actual', marker='o', color='blue', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('House Price')
    plt.title('Predicted vs Actual Prices (first 50 test samples)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'prediction_curve.png'))
    plt.close()

    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    # argparse lets you (or TA) run with custom settings from command line
    # example: python task1_linear_regression.py --lr 0.001 --epochs 200
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./Task-1 Dataset (California Housing)/california_housing_train.csv")
    parser.add_argument("--test_path", type=str, default="./Task-1 Dataset (California Housing)/california_housing_test.csv")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--init_type", type=str, default="he", choices=["he", "random", "zeros"])
    parser.add_argument("--run_experiments", action="store_true")
    args = parser.parse_args()

    # load data
    df_train, df_test = load_data(args.train_path, args.test_path)

    # separate features (X) from target price (Y)
    X, Y = separate_features_target(df_train)
    test_X, test_Y = separate_features_target(df_test)

    # split training data into train (85%) + validation (15%)
    train_X, train_Y, val_X, val_Y = split_data(X, Y)

    # compute mean and std from training data only
    mean = compute_mean(train_X)
    std = compute_std(train_X)

    if args.run_experiments:
        # ---- EXPERIMENT MODE ----
        # runs all 10 experiments the assignment asks for and saves results to results/task1/

        # keep raw (unnormalized) copies for "without normalization" experiments
        train_X_raw = train_X.copy()
        val_X_raw = val_X.copy()
        test_X_raw = test_X.copy()

        # create normalized copies
        train_X_norm = normalize(train_X, mean, std)
        val_X_norm = normalize(val_X, mean, std)
        test_X_norm = normalize(test_X, mean, std)

        # each experiment: (learning_rate, epochs, init_type, hidden_neurons, use_normalization)
        experiments = [
            (0.01,    100, "he",     8, True),    # exp1: baseline - good lr, he init
            (0.00001, 100, "he",     8, True),    # exp2: lr too small - barely learns
            (0.9,     100, "he",     8, True),    # exp3: lr too big - explodes
            (0.01,    150, "he",     8, True),    # exp4: more epochs
            (0.01,    100, "random", 8, True),    # exp5: random init instead of he
            (0.01,    100, "zeros",  8, True),    # exp6: zero init - cant learn
            (0.01,    100, "he",     2, True),    # exp7: only 2 hidden neurons
            (0.01,    150, "he",     2, True),    # exp8: 2 hidden + more epochs
            (0.01,    100, "he",     8, False),   # exp9: no normalization, 8 hidden
            (0.01,    100, "he",     2, False),   # exp10: no normalization, 2 hidden
        ]

        results = []

        for i, (lr, epochs, init, hidden, use_norm) in enumerate(experiments):
            norm_label = "norm" if use_norm else "raw"
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}: hidden={hidden}, init={init}, lr={lr}, epochs={epochs}, norm={use_norm}")
            print(f"{'='*60}")

            # pick normalized or raw data depending on the experiment
            if use_norm:
                tx, vx, tex = train_X_norm, val_X_norm, test_X_norm
            else:
                tx, vx, tex = train_X_raw, val_X_raw, test_X_raw

            # create fresh dataloader and network for each experiment
            dataset = HousingDataset(tx, train_Y)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            net = LinearRegressionNetwork(input_size=8, hidden_size=hidden, init_type=init)

            # train the model
            net, t_losses, v_losses, t_r2, v_r2 = train(net, loader, tx, train_Y, vx, val_Y, epochs, lr)

            # test on unseen data
            test_loss, test_r2, test_pred = test(net, tex, test_Y)

            # save plots in a separate folder for this experiment
            # eg: results/task1/exp1_h8_he_lr0.01_ep100_norm/
            exp_folder = f"./results/task1/exp{i+1}_h{hidden}_{init}_lr{lr}_ep{epochs}_{norm_label}"
            plot_results(t_losses, v_losses, t_r2, v_r2, test_loss, test_r2, test_pred, test_Y, exp_folder)

            # store results so we can print a table at the end
            results.append({
                'hidden': hidden, 'init': init, 'lr': lr, 'epochs': epochs,
                'norm': use_norm, 'mean_val_loss': np.mean(v_losses),
                'test_loss': test_loss, 'test_r2': test_r2
            })

        # print a comparison table (copy this into your report)
        print(f"\n{'='*100}")
        print("RESULTS TABLE")
        print(f"{'='*100}")
        print(f"{'#':<4} {'Hidden':<8} {'Init':<8} {'LR':<10} {'Epochs':<8} {'Norm':<6} {'Mean Val Loss':<15} {'Test Loss':<12} {'Test R2':<10}")
        print("-" * 100)
        for i, r in enumerate(results):
            print(f"{i+1:<4} {r['hidden']:<8} {r['init']:<8} {r['lr']:<10} {r['epochs']:<8} {str(r['norm']):<6} {r['mean_val_loss']:<15.4f} {r['test_loss']:<12.4f} {r['test_r2']:<10.4f}")

        # find which experiment had the lowest validation loss
        best = min(results, key=lambda x: x['mean_val_loss'])
        print(f"\nBest: hidden={best['hidden']}, init={best['init']}, lr={best['lr']}, epochs={best['epochs']}")

    else:
        # ---- SINGLE RUN MODE ----
        # just train once with the settings you passed via command line

        # normalize the data
        train_X = normalize(train_X, mean, std)
        val_X = normalize(val_X, mean, std)
        test_X = normalize(test_X, mean, std)

        # create dataloader and network
        dataset = HousingDataset(train_X, train_Y)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        net = LinearRegressionNetwork(input_size=8, hidden_size=args.hidden_size, init_type=args.init_type)

        # train
        print("Starting training...")
        net, t_losses, v_losses, t_r2, v_r2 = train(net, loader, train_X, train_Y, val_X, val_Y, args.epochs, args.lr)
        print("Training complete!")

        # test on unseen data
        test_loss, test_r2, test_pred = test(net, test_X, test_Y)

        # save trained model
        save_model(net, mean, std, './saved_models/linear_model.pkl')

        # save plots in its own folder based on the settings used
        run_folder = f"./results/task1/single_h{args.hidden_size}_{args.init_type}_lr{args.lr}_ep{args.epochs}"
        plot_results(t_losses, v_losses, t_r2, v_r2, test_loss, test_r2, test_pred, test_Y, run_folder)

