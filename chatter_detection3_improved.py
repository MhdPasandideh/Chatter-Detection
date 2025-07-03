import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from scipy.fft import fft
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set base directory
BASE_DIR = r"C:\Users\Lenovo\Dropbox\PC\Desktop\DBSCAN Clustring"
MILL_FILE = os.path.join(BASE_DIR, "mill.mat")
RESULTS_FILE = os.path.join(BASE_DIR, 'chatter_detection_results.png')
os.makedirs(BASE_DIR, exist_ok=True)

# Set device for PyTorch (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    """PyTorch Autoencoder implementation with flexible architecture"""
    def __init__(self, input_dim: int, encoding_dim: int = 2, hidden_dims: Tuple[int, int] = (64, 32)):
        """
        Initialize the Autoencoder
        
        Args:
            input_dim: Dimension of input features
            encoding_dim: Dimension of the latent space (default: 2)
            hidden_dims: Tuple of hidden layer dimensions (default: (64, 32))
        """
        super(Autoencoder, self).__init__()
        
        # Validate dimensions
        if not isinstance(hidden_dims, tuple) or len(hidden_dims) != 2:
            raise ValueError("hidden_dims must be a tuple of length 2")
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], encoding_dim),
            nn.ReLU()
        )
        
        # Decoder layers (mirror architecture)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input data to latent space"""
        return self.encoder(x)

def load_mill_data() -> pd.DataFrame:
    """
    Load the milling dataset from .mat file with robust error handling
    
    Returns:
        pd.DataFrame: Loaded data with sensor measurements and time
        
    Raises:
        FileNotFoundError: If the .mat file is not found
        ValueError: If the .mat file structure is invalid
    """
    try:
        if not os.path.exists(MILL_FILE):
            raise FileNotFoundError(f"MATLAB file not found at {MILL_FILE}")
        
        mat_data = loadmat(MILL_FILE)
        print("Available keys in the .mat file:", list(mat_data.keys()))
        
        # Attempt to extract relevant data - adjust these keys based on your actual data structure
        vibration_data = mat_data.get('vibration', None)
        if vibration_data is None:
            raise ValueError("'vibration' key not found in .mat file")
            
        # Get time vector or create one if not available
        time_vector = mat_data.get('time', np.arange(vibration_data.shape[0]))
        
        # Ensure we have 6 channels of data as expected
        if vibration_data.shape[1] != 6:
            raise ValueError(f"Expected 6 channels of data, got {vibration_data.shape[1]}")
        
        # Create DataFrame with proper column names
        data = pd.DataFrame({
            'Current_x': vibration_data[:, 0],
            'Current_y': vibration_data[:, 1],
            'Current_s': vibration_data[:, 2],
            'Abs_Position': vibration_data[:, 3],
            'ServoSpindleSpeed': vibration_data[:, 4],
            'ServoSpindleLoad': vibration_data[:, 5],
            'Time': time_vector.flatten()
        })
        
        print("Successfully loaded data with shape:", data.shape)
        return data
        
    except Exception as e:
        print(f"Error loading .mat file: {str(e)}")
        print("Creating synthetic data as fallback...")
        return create_synthetic_data()

def create_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create synthetic CNC machine data with chatter events
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic CNC machine data
    """
    print(f"Generating synthetic data with {n_samples} samples...")
    
    time = np.linspace(0, 10, n_samples)
    
    # Normal operation signals
    current_x = np.sin(2 * np.pi * 5 * time) + 0.1 * np.random.normal(size=n_samples)
    current_y = np.sin(2 * np.pi * 5 * time + np.pi/4) + 0.1 * np.random.normal(size=n_samples)
    current_s = np.sin(2 * np.pi * 3 * time) + 0.1 * np.random.normal(size=n_samples)
    abs_position = time * 10 + np.random.normal(0, 0.5, size=n_samples)
    servo_spindle_speed = 1000 + 100 * np.sin(2 * np.pi * 0.5 * time) + np.random.normal(0, 10, size=n_samples)
    servo_spindle_load = 50 + 20 * np.sin(2 * np.pi * 0.5 * time) + np.random.normal(0, 5, size=n_samples)
    
    # Add chatter events (anomalies)
    n_chatter = max(50, int(n_samples * 0.05))  # 5% of samples or 50, whichever is larger
    chatter_indices = np.random.choice(n_samples, size=n_chatter, replace=False)
    
    # Add more significant deviations during chatter events
    current_x[chatter_indices] += 0.5 * np.random.normal(size=n_chatter)
    current_y[chatter_indices] += 0.5 * np.random.normal(size=n_chatter)
    current_s[chatter_indices] += 0.3 * np.random.normal(size=n_chatter)
    servo_spindle_speed[chatter_indices] += 30 * np.random.normal(size=n_chatter)
    servo_spindle_load[chatter_indices] += 15 * np.random.normal(size=n_chatter)
    
    return pd.DataFrame({
        'Current_x': current_x,
        'Current_y': current_y,
        'Current_s': current_s,
        'Abs_Position': abs_position,
        'ServoSpindleSpeed': servo_spindle_speed,
        'ServoSpindleLoad': servo_spindle_load,
        'Time': time
    })

def preprocess_data(data: pd.DataFrame, method: str = 'normalization') -> Tuple[np.ndarray, object]:
    """
    Preprocess the data using specified scaling method
    
    Args:
        data: Input DataFrame with raw signals
        method: Scaling method ('normalization' or 'standardization')
        
    Returns:
        Tuple of (scaled_data, scaler_object)
        
    Raises:
        ValueError: If invalid preprocessing method is specified
    """
    if method not in ['normalization', 'standardization']:
        raise ValueError(f"Invalid preprocessing method: {method}. Choose 'normalization' or 'standardization'")
    
    signals = data[['Current_x', 'Current_y', 'Current_s', 
                   'Abs_Position', 'ServoSpindleSpeed', 'ServoSpindleLoad']]
    
    print(f"Applying {method} to data...")
    
    if method == 'normalization':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    scaled_data = scaler.fit_transform(signals)
    return scaled_data, scaler

def apply_fft(signals: np.ndarray) -> np.ndarray:
    """Apply Fast Fourier Transform to signals along time axis"""
    return np.abs(fft(signals, axis=0))

def segment_data(data: np.ndarray, window_size: int = 100, overlap: float = 0.5) -> np.ndarray:
    """
    Segment time-series data into overlapping windows
    
    Args:
        data: Input time-series data
        window_size: Number of samples per window
        overlap: Fraction of overlap between windows (0 to 1)
        
    Returns:
        np.ndarray: Segmented data with shape (n_windows, window_size, n_features)
    """
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in range [0, 1)")
    
    step_size = int(window_size * (1 - overlap))
    n_segments = (len(data) - window_size) // step_size + 1
    segments = np.zeros((n_segments, window_size, data.shape[1]))
    
    for i in range(n_segments):
        start = i * step_size
        segments[i] = data[start:start+window_size]
    
    return segments

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.001,
    epochs: int = 100,
    patience: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train the autoencoder model with early stopping
    
    Args:
        model: Autoencoder model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        learning_rate: Learning rate for optimizer
        epochs: Maximum number of epochs
        patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Tuple of (train_loss_history, val_loss_history)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    progress_bar = tqdm(range(epochs), desc="Training Autoencoder")
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Train Loss': f"{train_loss:.4f}",
            'Val Loss': f"{val_loss:.4f}"
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    return np.array(train_losses), np.array(val_losses)

def evaluate_reconstruction(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Evaluate reconstruction performance on given data
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader for evaluation data
        
    Returns:
        Average reconstruction loss (MSE)
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def hyperparameter_tuning(
    X_train: np.ndarray,
    X_val: np.ndarray,
    param_grid: Dict[str, list],
    n_epochs: int = 50
) -> Tuple[nn.Module, Dict, float]:
    """
    Perform hyperparameter tuning for the autoencoder
    
    Args:
        X_train: Training data
        X_val: Validation data
        param_grid: Dictionary of hyperparameters to search
        n_epochs: Number of epochs for each configuration
        
    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    print("\nStarting hyperparameter tuning...")
    total_combinations = len(ParameterGrid(param_grid))
    print(f"Testing {total_combinations} parameter combinations")
    
    for i, params in enumerate(ParameterGrid(param_grid), 1):
        print(f"\n[{i}/{total_combinations}] Testing: {params}")
        
        # Create data loaders with current batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size']
        )
        
        # Initialize model with current parameters
        model = Autoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=params['encoding_dim']
        ).to(device)
        
        # Train model
        train_losses, val_losses = train_autoencoder(
            model,
            train_loader,
            val_loader,
            learning_rate=params['learning_rate'],
            epochs=n_epochs
        )
        
        # Evaluate on validation set
        val_loss = val_losses[-1]  # Use final validation loss
        
        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_model = model
            print(f"New best score: {best_score:.4f}")
    
    print("\nHyperparameter tuning completed!")
    print(f"Best params: {best_params}")
    print(f"Best validation loss: {best_score:.4f}")
    
    return best_model, best_params, best_score

def apply_dbscan(encoded_data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Apply DBSCAN clustering to encoded data
    
    Args:
        encoded_data: Data in latent space
        eps: DBSCAN neighborhood radius
        min_samples: Minimum points to form a cluster
        
    Returns:
        Array of cluster labels
    """
    print(f"\nApplying DBSCAN with eps={eps}, min_samples={min_samples}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(encoded_data)

def evaluate_clustering(encoded_data: np.ndarray, clusters: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate clustering performance
    
    Args:
        encoded_data: Data in latent space
        clusters: Cluster labels
        
    Returns:
        Tuple of (silhouette_score, davies_bouldin_score)
    """
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    print(f"\nEvaluating clustering with {n_clusters} clusters...")
    
    if n_clusters < 2:
        print("Warning: Not enough clusters for meaningful evaluation")
        return -1, float('inf')
    
    silhouette = silhouette_score(encoded_data, clusters)
    db_index = davies_bouldin_score(encoded_data, clusters)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Index: {db_index:.3f}")
    
    return silhouette, db_index

def plot_results(
    data: pd.DataFrame,
    encoded_data: np.ndarray,
    clusters: np.ndarray,
    train_loss: Optional[np.ndarray] = None,
    val_loss: Optional[np.ndarray] = None,
    save_path: str = RESULTS_FILE
):
    """
    Create comprehensive visualization of results
    
    Args:
        data: Original time-series data
        encoded_data: Encoded latent space data
        clusters: Cluster assignments
        train_loss: Training loss history
        val_loss: Validation loss history
        save_path: Path to save the figure
    """
    plt.figure(figsize=(20, 16))
    plt.suptitle("Chatter Detection Analysis Results", fontsize=16, y=1.02)
    
    # Plot 1: Original Signals
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
    for col in data.columns[:-1]:  # Exclude 'Time'
        ax1.plot(data['Time'], data[col], label=col)
    ax1.set_title('Original CNC Signals')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Value')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Plot 2: FFT Analysis
    ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
    fft_data = apply_fft(data[data.columns[:-1]].values)
    for i, col in enumerate(data.columns[:-1]):
        ax2.plot(fft_data[:, i], label=col)
    ax2.set_title('Frequency Spectrum (FFT)')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    
    # Plot 3: Latent Space Clustering
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=1)
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        if cluster == -1:
            # Noise points
            ax3.scatter(
                encoded_data[clusters == cluster, 0],
                encoded_data[clusters == cluster, 1],
                c='k', marker='x', label='Noise', alpha=0.5
            )
        else:
            ax3.scatter(
                encoded_data[clusters == cluster, 0],
                encoded_data[clusters == cluster, 1],
                c=[color], label=f'Cluster {cluster}', alpha=0.7
            )
    
    ax3.set_title('Latent Space Clustering with DBSCAN')
    ax3.set_xlabel('Encoded Dimension 1')
    ax3.set_ylabel('Encoded Dimension 2')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Training History (if provided)
    if train_loss is not None and val_loss is not None:
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2, rowspan=1)
        ax4.plot(train_loss, label='Training Loss', linewidth=2)
        ax4.plot(val_loss, label='Validation Loss', linewidth=2)
        ax4.set_title('Autoencoder Training History')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MSE Loss')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nResults saved to {save_path}")
    plt.show()

def main():
    print("\n=== CNC Machine Chatter Detection Pipeline ===")
    print(f"Using device: {device}")
    
    try:
        # Step 1: Load and prepare data
        print("\n1. Loading and preparing data...")
        data = load_mill_data()
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        scaled_data, scaler = preprocess_data(data, method='normalization')
        
        # Step 3: Segment data
        print("\n3. Segmenting time-series data...")
        segmented_data = segment_data(scaled_data, window_size=100, overlap=0.5)
        X = segmented_data.reshape(segmented_data.shape[0], -1)  # Flatten windows
        
        # Split into train and test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Step 4: Autoencoder hyperparameter tuning
        print("\n4. Performing hyperparameter tuning...")
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'encoding_dim': [2, 4]
        }
        
        best_model, best_params, best_score = hyperparameter_tuning(
            X_train, X_test, param_grid, n_epochs=50
        )
        
        # Step 5: Train final autoencoder with best params
        print("\n5. Training final autoencoder...")
        # Prepare data loaders with best batch size
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=best_params['batch_size']
        )
        
        # Initialize and train final model
        final_model = Autoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=best_params['encoding_dim']
        ).to(device)
        
        train_loss, val_loss = train_autoencoder(
            final_model,
            train_loader,
            test_loader,
            learning_rate=best_params['learning_rate'],
            epochs=100
        )
        
        # Step 6: Encode data
        print("\n6. Encoding data...")
        final_model.eval()
        with torch.no_grad():
            encoded_train = final_model.encode(X_train_tensor).cpu().numpy()
            encoded_test = final_model.encode(X_test_tensor).cpu().numpy()
        
        # Step 7: Cluster encoded data with DBSCAN
        print("\n7. Applying DBSCAN clustering...")
        clusters_train = apply_dbscan(encoded_train, eps=0.3, min_samples=5)
        clusters_test = apply_dbscan(encoded_test, eps=0.3, min_samples=5)
        
        # Step 8: Evaluate clustering
        print("\n8. Evaluating clustering performance...")
        print("\nTraining set evaluation:")
        silhouette_train, db_index_train = evaluate_clustering(encoded_train, clusters_train)
        
        print("\nTest set evaluation:")
        silhouette_test, db_index_test = evaluate_clustering(encoded_test, clusters_test)
        
        # Step 9: Visualize results
        print("\n9. Visualizing results...")
        # Use a subset for visualization
        vis_data = data.iloc[:2000]  # First 2000 samples for clearer visualization
        vis_scaled, _ = preprocess_data(vis_data, method='normalization')
        vis_segments = segment_data(vis_scaled, window_size=100, overlap=0.5)
        vis_X = vis_segments.reshape(vis_segments.shape[0], -1)
        
        with torch.no_grad():
            vis_encoded = final_model.encode(torch.FloatTensor(vis_X).to(device)).cpu().numpy()
        
        vis_clusters = apply_dbscan(vis_encoded, eps=0.3, min_samples=5)
        
        plot_results(
            vis_data,
            vis_encoded,
            vis_clusters,
            train_loss,
            val_loss
        )
        
        print("\n=== Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"\n!!! Pipeline failed with error: {str(e)}")
        print("Please check the error message and ensure:")
        print("1. The MATLAB file exists and is accessible")
        print("2. You have all required packages installed")
        print("3. Your system meets the hardware requirements")
        raise

if __name__ == "__main__":
    main()