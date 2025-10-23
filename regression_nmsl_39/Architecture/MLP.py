import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split  # for automatic validation splitting

class MLPRegressorPyTorch(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=False,
                 momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                 dropout_rate=0.2):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.dropout_rate = dropout_rate

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Uncomment the following to force CPU usage:
        # self.device = torch.device("cpu")

    def _build_model(self, input_dim, output_dim):
        layers = []
        layer_sizes = [input_dim] + list(self.hidden_layer_sizes) + [output_dim]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                # Add activation function.
                if self.activation == 'relu':
                    layers.append(nn.ReLU())
                elif self.activation == 'tanh':
                    layers.append(nn.Tanh())
                elif self.activation == 'logistic':
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError("Unsupported activation function.")
                # Optionally add dropout if dropout_rate > 0.
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(p=self.dropout_rate))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        print("Device: ", self.device)
        # Automatic validation split if early stopping is enabled but no validation set is provided.
        if self.early_stopping and (X_val is None or y_val is None) and self.validation_fraction is not None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state
            )
        
        # Convert training data to tensors.
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Convert validation data if provided.
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)
        
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        # Build or reuse the model based on warm_start flag.
        if not self.warm_start or self.model is None:
            self.model = self._build_model(input_dim, output_dim).to(self.device)
            # Xavier (Glorot) initialization for Linear layers.
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            self.model.apply(init_weights)
        
        # Configure the optimizer.
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2), eps=self.epsilon, weight_decay=self.alpha
            )
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate_init, momentum=self.momentum,
                nesterov=self.nesterovs_momentum, weight_decay=self.alpha
            )
        else:
            raise ValueError("Unsupported solver.")
        
        self.criterion = nn.MSELoss().to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Fix 1: Use a reasonable batch size instead of 'auto'
        if self.batch_size == 'auto':
            # Use a reasonable default batch size (32 or 64) instead of the entire dataset
            batch_size = min(64, len(X))
        else:
            batch_size = self.batch_size
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)
        
        best_loss = float('inf')
        no_improve_count = 0
        best_model_state = None
        
        for epoch in range(self.max_iter):
            self.model.train()  # ensure the model is in training mode
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Fix 2: Ensure data stays on GPU (redundant but explicit)
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            epoch_loss /= len(dataloader.dataset)
            
            # Evaluate on validation set if available.
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val)
                    val_loss = self.criterion(val_predictions, y_val).item()
            else:
                val_loss = None
            
            if self.verbose:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Training Loss: {epoch_loss:.6f}")
            
            # Early stopping
            if self.early_stopping and val_loss is not None:
                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    no_improve_count = 0
                    best_model_state = self.model.state_dict()
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break
        
        # Optionally reload the best model obtained during training.
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self
    
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Fix 3: Process in batches for large datasets to avoid memory issues
        if len(X) > 100000:  # Arbitrary threshold, adjust based on your GPU memory
            batch_size = 10000
            dataset = torch.utils.data.TensorDataset(X)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_X = batch[0]
                    batch_pred = self.model(batch_X)
                    predictions.append(batch_pred)
            
            predictions = torch.cat(predictions, dim=0).cpu().numpy()
        else:
            with torch.no_grad():
                predictions = self.model(X).cpu().numpy()
            
        return predictions

if __name__ == '__main__':
    # Synthetic data
    X_train = torch.randn(100000, 10)
    y_train = torch.randn(100000, 1)
    X_val = torch.randn(10000, 10)
    y_val = torch.randn(10000, 1)
    mlp = MLPRegressorPyTorch()
    mlp.fit(X_train, y_train, X_val, y_val)
    predictions = mlp.predict(X_val)
    print(predictions)