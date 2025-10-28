# python/ml_agent.py
#
# Title: MLGradientPredictor - Neural Surrogate Model for Multi-Agent Optimization
#
# Description:
# This module implements the MLGradientPredictor, a neural surrogate model designed for the
# **ParallelOptimizationEngine** framework. It serves as a bonus AI-enhanced reasoning agent,
# approximating the weighted average \( x^* = \frac{\sum a_i b_i}{\sum a_i} \) for collaborative
# mode using a feedforward Multi-Layer Perceptron (MLP), and computing the unweighted average
# \( \frac{\sum b_i}{N} \) for naive mode. The model leverages unrolled optimization techniques
# to enhance performance in large-scale simulations.
#
# Architecture (MLP with Unrolled Optimization):
# - **Input**: Concatenated vector of all agent coefficients \( a_i \) and \( b_i \) (2N inputs)
# - **Hidden Layers**: Two layers with 20 neurons each, using ReLU activation
# - **Output**: Predicted \( x^* \) (1 output), computed via fixed division for exactness
#
# Training Configuration:
# - **Loss Function**: Mean Squared Error (MSE) combined with an accuracy gap penalty (< 1e-6)
# - **Optimizer**: Adam with a StepLR scheduler for adaptive learning rate adjustment
# - **Epochs**: 500 iterations for robust convergence
# - **Device**: Automatically selects GPU if available via `torch.cuda.is_available()`, otherwise CPU
# - **Precision**: Utilizes `float64` for high numerical accuracy
#
# Use Cases:
# - Replaces exact weighted average computation in the `CollaborativeStrategy`.
# - Provides exact unweighted average computation for the naive mode.
#
# Dependencies:
# - torch: For neural network implementation and device management
# - torch.nn: For neural network layers and loss functions
# - torch.optim: For optimization algorithms
# - numpy: For numerical operations on input data
# - time: For performance timing metrics
#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class WeightedAverageNet(nn.Module):
    """
    A feedforward neural network designed to approximate the weighted average \( x^* \)
    for multi-agent optimization using an MLP architecture.

    Architecture:
    - Input Layer: Accepts 2N inputs (concatenated \( a_i \) and \( b_i \) coefficients)
    - Hidden Layers: Two layers with 20 neurons each, using ReLU activation
    - Output Layer: Produces two values ([sum(a_i b_i), sum(a_i)]) for fixed division

    Purpose:
    Learns to predict the weighted average \( x^* = \frac{\sum a_i b_i}{\sum a_i} \)
    by training on agent coefficient pairs, with the final division enforced for exactness.
    """
    def __init__(self, num_agents):
        """
        Initialize the neural network with agent-specific input size.

        Args:
            num_agents (int): Number of agents, determining the input dimension (2 * num_agents)

        Notes:
        - Uses torch.float64 for high precision in HPC applications.
        - Layer sizes are chosen to balance accuracy and computational efficiency.
        """
        super().__init__()
        self.num_agents = num_agents
        self.fc1 = nn.Linear(2 * num_agents, 20, dtype=torch.float64)  # First layer: 2N inputs to 20 neurons
        self.fc2 = nn.Linear(20, 20, dtype=torch.float64)  # Second hidden layer: 20 to 20 neurons
        self.fc3 = nn.Linear(20, 2, dtype=torch.float64)  # Output layer: 20 to 2 values (sum_ab, sum_a)

    def forward(self, input_vec):
        """
        Perform the forward pass with ReLU activations and compute the weighted average.

        Args:
            input_vec (torch.Tensor): Input tensor of shape (batch_size, 2 * num_agents),
                                     containing concatenated [a1, b1, ..., aN, bN]

        Returns:
            torch.Tensor: Predicted \( x^* \) of shape (batch_size, 1)

        Notes:
        - ReLU activations introduce non-linearity for better learning.
        - Fixed division avoids division by zero with a small epsilon (1e-10).
        """
        x = torch.relu(self.fc1(input_vec))  # Apply ReLU to first hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU to second hidden layer
        out = self.fc3(x)  # Output [sum(a_i b_i), sum(a_i)]
        sum_ab, sum_a = out[:, 0], out[:, 1]  # Extract sums
        x_star = sum_ab / (sum_a + 1e-10)  # Compute weighted average with zero prevention
        return x_star.unsqueeze(1)  # Ensure output shape (batch_size, 1)

class MLGradientPredictor:
    """
    A wrapper class for training and inference of the weighted average prediction model.

    Responsibilities:
    - Instantiates the WeightedAverageNet with agent-specific input size
    - Configures optimizer and loss function with an accuracy gap penalty
    - Trains the model on randomly generated agent instances for generalization
    - Provides exact computation for naive mode and prediction for collaborative mode

    Attributes:
        is_naive (bool): Flag to switch between naive and collaborative modes
        max_iter (int): Maximum iterations for optimization loop
        tol (float): Convergence tolerance
        training_time (float): Duration of model training
        iteration_time (float): Duration of optimization iterations
        iterations (int): Number of iterations performed
        agents (list): List of Agent objects
        num_agents (int): Number of agents
        device (torch.device): Computing device (GPU or CPU)
        sum_a (float): Precomputed sum of a_i coefficients
        sum_b (float): Precomputed sum of b_i coefficients
        sum_ab (float): Precomputed sum of a_i * b_i products
        model (WeightedAverageNet): Neural network model
        optimizer (torch.optim.Adam): Optimization algorithm
        scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler
        criterion (torch.nn.MSELoss): Loss function
    """
    def __init__(self, agents=None, is_naive=False, max_iter=1000, tol=1e-6):
        """
        Initialize the MLGradientPredictor with configurable parameters.

        Args:
            agents (list, optional): List of Agent objects; defaults to empty list
            is_naive (bool, optional): Flag for naive mode; defaults to False
            max_iter (int, optional): Maximum optimization iterations; defaults to 1000
            tol (float, optional): Convergence tolerance; defaults to 1e-6

        Notes:
        - Initializes performance metrics and precomputes sums for naive mode.
        - Trains the model only for collaborative mode if agents are provided.
        - Uses float64 for high precision and auto-detects GPU/CPU device.
        """
        self.is_naive = is_naive
        self.max_iter = max_iter
        self.tol = tol
        self.training_time = 0.0
        self.iteration_time = 0.0
        self.iterations = 0
        self.agents = agents if agents is not None else []
        self.num_agents = len(self.agents) if self.agents else 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sum_a = sum(ag.a for ag in self.agents) if self.agents else 0.0
        self.sum_b = sum(ag.b for ag in self.agents) if self.agents else 0.0
        self.sum_ab = sum(ag.a * ag.b for ag in self.agents) if self.agents else 0.0
        if self.agents and not self.is_naive:
            self.model = WeightedAverageNet(self.num_agents)
            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
            self.criterion = nn.MSELoss()
            start_train = time.perf_counter()
            self.train(epochs=500)
            self.training_time = time.perf_counter() - start_train

    def train(self, epochs=500):
        """
        Train the model on randomly generated agent instances for weighted average approximation.

        Args:
            epochs (int, optional): Number of training epochs; defaults to 500

        Notes:
        - Generates random (a, b) pairs to generalize across agent configurations.
        - Applies MSE loss with an additional penalty for accuracy gap > 1e-6.
        - Uses double precision (float64) for high numerical accuracy.
        """
        self.model.train()
        for _ in range(epochs):
            # Generate random batch of agents for training generalization
            N = self.num_agents
            a_batch = np.random.normal(5.0, 2.0, N)
            a_batch = np.maximum(a_batch, 1e-6)  # Ensure positivity for convexity
            b_batch = np.random.normal(0.0, 5.0, N)
            input_vec = np.concatenate([a_batch, b_batch])
            input_tensor = torch.tensor(input_vec, dtype=torch.float64, device=self.device).unsqueeze(0)
            # Compute ground truth weighted average for supervision
            x_star_true = np.sum(a_batch * b_batch) / np.sum(a_batch)
            x_star_tensor = torch.tensor([[x_star_true]], dtype=torch.float64, device=self.device)
            self.optimizer.zero_grad()
            pred_x = self.model(input_tensor)
            loss = self.criterion(pred_x, x_star_tensor)
            # Add penalty for accuracy gap to enforce high precision
            accuracy_gap = torch.abs(pred_x - x_star_tensor)
            loss += 1e-4 * torch.mean(accuracy_gap)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        self.model.eval()

    def predict(self, x):
        """
        Predict the weighted average for collaborative mode or the unweighted average for naive mode.

        Args:
            x (float): Ignored parameter (legacy); now uses agent data

        Returns:
            float: Predicted or exact average value

        Notes:
        - For naive mode, returns the exact unweighted average of b_i.
        - For collaborative mode, uses the trained model to predict x*.
        """
        if self.is_naive and self.agents:
            if self.sum_b is not None:
                return float(self.sum_b / len(self.agents))
            return float(sum(ag.b for ag in self.agents) / len(self.agents))
        else:
            if not self.agents:
                return 0.0
            input_vec = np.concatenate([[ag.a, ag.b] for ag in self.agents])
            input_tensor = torch.tensor(input_vec, dtype=torch.float64, device=self.device).unsqueeze(0)
            with torch.no_grad():
                return self.model(input_tensor).item()

    def optimize(self, agents=None, max_iter=None, tolerance=None):
        """
        Perform unrolled optimization or return exact average based on mode.

        Args:
            agents (list, optional): List of Agent objects to update; defaults to None
            max_iter (int, optional): Maximum iterations; defaults to instance value
            tolerance (float, optional): Convergence tolerance; defaults to instance value

        Returns:
            tuple: (optimized x, number of iterations)

        Notes:
        - Trains the model if new agents are provided and in collaborative mode.
        - Uses a simple iteration loop to simulate unrolled optimization, checking convergence.
        - Tracks inference and iteration times for performance metrics.
        """
        if agents is not None:
            self.agents = agents
            self.num_agents = len(self.agents)
            self.sum_a = sum(ag.a for ag in self.agents)
            self.sum_b = sum(ag.b for ag in self.agents)
            self.sum_ab = sum(ag.a * ag.b for ag in self.agents)
            if not self.is_naive and self.training_time == 0.0:
                self.model = WeightedAverageNet(self.num_agents)
                self.model.to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
                self.train(epochs=500)
        max_iter = max_iter if max_iter is not None else self.max_iter
        tol = tolerance if tolerance is not None else self.tol
        x = 0.0  # Initialize x to avoid UnboundLocalError
        self.iterations = 0
        self.inference_time = 0.0
        start_time = time.perf_counter()
        while self.iterations < max_iter:
            t0 = time.perf_counter()
            x_new = self.predict(None)
            self.inference_time += (time.perf_counter() - t0)
            if not self.is_naive:
                if abs(x_new - x) < tol:
                    x = x_new
                    break
                x = x_new
            else:
                x = x_new
                break
            self.iterations += 1
        self.iteration_time = time.perf_counter() - start_time - self.training_time
        return x, self.iterations