import jax
import jax.numpy as jnp
from jax import random, jit
import flax.linen as nn
import optax
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Hyperparameters
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Data Preprocessing using sklearn's fetch_openml
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'].to_numpy(), mnist['target'].astype(int).to_numpy()  # Convert to numpy arrays
    X = X / 255.0  # Normalize to [0, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return jax.device_put((X_train, y_train)), jax.device_put((X_test, y_test))

# Model definition using Flax
class SimpleNN(nn.Module):
    hidden_size: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)  # Fully connected layer
        x = nn.relu(x)                      # ReLU activation
        x = nn.Dense(self.num_classes)(x)   # Output layer
        return x

# Initialize model parameters using Flax
def init_model(rng):
    model = SimpleNN(hidden_size=hidden_size, num_classes=num_classes)
    params = model.init(rng, jnp.ones((1, 784)))  # Dummy input for initialization
    return params['params'], model

# Cross-entropy loss
def cross_entropy_loss(params, model, x, y):
    logits = model.apply({'params': params}, x)
    one_hot_labels = jax.nn.one_hot(y, num_classes)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))

# Training step function
@jax.jit
def train_step(params, model, x, y, opt_state, optimizer):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# Accuracy computation
@jax.jit
def compute_accuracy(params, model, x, y):
    logits = model.apply({'params': params}, x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)

# Main training loop
def train_and_evaluate():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    # Initialize model and optimizer
    rng = random.PRNGKey(0)
    params, model = init_model(rng)  # Only initialize once with rng
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop
    num_batches = len(train_images) // batch_size
    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_idx = jax.random.randint(rng, (batch_size,), 0, len(train_images))
            batch_images = train_images[batch_idx]
            batch_labels = train_labels[batch_idx]

            # Train step
            params, opt_state, loss = train_step(params, model, batch_images, batch_labels, opt_state, optimizer)

        # Compute accuracy on test set
        accuracy = compute_accuracy(params, model, test_images, test_labels)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

# Run training
train_and_evaluate()
