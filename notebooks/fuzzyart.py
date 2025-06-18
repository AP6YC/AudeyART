"""
A module containing all of the pedagogical FuzzyART material.
"""

# STDLIB IMPORTS

# For manipulating local paths in an object-oriented way
from pathlib import Path

# 3RD PARTY IMPORTS

# The PyTorch library containing neural network utilities and the Tensor datatype
import torch
# A convenient import of Tensor so that we don't have to write torch.Tensor every time
from torch import Tensor
# Pandas for loading and manipulating data as a DataFrame
import pandas as pd
# Numpy for handling numpy arrays (i.e., matplotlib doesn't understand Tensor types, but it does know numpy.nparray)
import numpy as np

# A sklearn utility for handling normalization of data automatically
from sklearn.preprocessing import MinMaxScaler
# From scikit-learn, for casting the data to 2D for visualization.
# This is not how the data actually looks in 4D, but the best that we can do is to cast it to 2D such that relative distances are mostly maintained.
from sklearn.manifold import TSNE
# An sklearn utility for converting a list of text labels into unique integers
from sklearn.preprocessing import LabelEncoder

# The most common way of importing matplotlib for plotting in Python
from matplotlib import pyplot as plt
# For manipulating axis tick locations
from matplotlib import ticker

# An IPython magic syntax that tells matplotlib to plot in a new cell instead of a new window
# %matplotlib inline


class DataContainer():

    def __init__(self):
        self.data, self.data_cc = self.load_data()
        # The number of samples is in the first dimensions of the complement-coded data
        self.n_samples = self.data_cc.shape[0]

        # The original dimension of the data is half of the complement coded dimension, which we make sure is cast back to an int
        self.dim = int(self.data_cc.shape[1] / 2)
        self.dim_cc = self.data_cc.shape[1]
        return

    def load_data(self):
        # Point to the local data file
        datafile = Path("..", "data", "iris", "iris.data")
        # Read the data as a CSV, manually declaring the headers since the file doesn't have them
        data = pd.read_csv(datafile, names=["SL", "SW", "PL", "PW", "Label"])

        # Intialize the scalar and update the values in-place to be normalized between [0, 1]
        scaler = MinMaxScaler()
        data[["SL", "SW", "PL", "PW"]] = scaler.fit_transform(data[["SL", "SW", "PL", "PW"]])

        # Change the text labels to integer labels
        label_encoder = LabelEncoder()
        data["Label"] = label_encoder.fit_transform(data["Label"])

        # Shuffle the data
        np.random.seed(12345)
        data = data.sample(frac=1).reset_index(drop=True)

        # Complement code the data by pushing it into a Tensor
        data_cc = torch.Tensor(data[["SL", "SW", "PL", "PW"]].values)
        # and appending the vector [1-x] along the feature dimension
        data_cc = torch.cat((data_cc, 1 - data_cc), dim=1)
        # What we get is a list of 8-dimensional samples
        return data, data_cc


# Define a class that contains everything we need
class AudeyART():

    # The constructor will be where we pass the hyperparameters necessary for the FuzzyART module to work.
    def __init__(
        self,
        dim: int,           # The original dimension of the data
        rho: float = 0.6,     # The vigilance parameter in [0, 1]
        beta: float = 0.5,    # The learning rate in [0, 1]
        alpha: float = 0.001  # The choice parameter (close to zero)
    ):
        # The weights with be of shape [n_categories, 2 * n_dimensions] for complement coded samples
        # NOTE: we start with weights of size [0, 8] with an uninitialized number of categories but known complement-coded dimension length
        self.W = Tensor(size=[0, 2*dim])
        # Keep track of the original dimension for later
        self.dim = dim
        # Save the rest of the operational hyperaparameters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        # self.map = {}
        self.map = []

        return

    # This function defines how we add a new category to the weight matrix
    def grow(self, x: Tensor):
        # This is a commented print statement for debugging the shapes of everything, kept here for reference
        # print(self.W.shape, x.shape, x.reshape((1, self.dim*2)).shape)

        # Growing the weight matrix ultimately means appending a new weight vector.
        # This is slow how it is written because the memory is overwritten each time we add a new weight, but it works fine for small datasets.
        self.W = torch.cat((self.W, x.reshape((1, self.dim*2))))

        return

    # This function defines what it means to initialize a new category and add it to the weight matrix
    def init_cat(self, x: Tensor):
        # First, we infer the existing number of categories
        n_categories = self.W.shape[0]
        # Next we initialize the "uncommitted node" with all ones
        new_W = torch.ones(2*self.dim)
        # We append the new node to the weights via our grow function
        self.grow(new_W)
        # Then we immediately update that category
        # NOTE: Python is 0-indexed, so the n_categories we got before we appended a new one happens to be the correct index to update
        self.learn(x, n_categories)

        return

    # This is the activation function to compute `T` given sample `x` and the weight at index `j`
    def activation(self, x: Tensor, j: int):
        # We will do this step-by-step to illustrate each computation.
        # First, get the element-wise minimum (fuzzy intersection) of weight `j` and sample `x`.
        xinw = torch.minimum(x, self.W[j, :])
        # Taking the 1-norm is simply a sum
        xinwnorm = torch.sum(xinw)
        # We also need the 1-norm of the weight on its own
        wnorm = torch.sum(self.W[j, :])
        # We compute the activation as the norm of the fuzzy intersection over the weight norm plus the choice parameter (to not divide by zero).
        Tj = xinwnorm / (self.alpha + wnorm)
        # The output is then just this one activation `T` for weight `j`
        return Tj

    # This is the match function to compute `M` given sample `x` and the weight at index `j`
    def match(self, x: Tensor, j: int):
        # Again, the fuzzy intersection is just the elementwise-minimum between sample `x` and weight `j`
        xinw = torch.minimum(x, self.W[j, :])
        # The 1-norm is simply a sum of all elements
        xinwnorm = torch.sum(xinw)
        # The match is defined as fuzzy intersection over the 1-norm of the sample, but we know that the sample is normalized to [0, 1] and complement coded, so that term will always be equal to the original dimension number
        Mj = xinwnorm / self.dim
        # The output here is the match value for weight `j`
        return Mj

    # This function describes what it means to update a winning weight at index `j` with sample `x`
    def learn(self, x: Tensor, j: int):
        # We accidentally wrote the wrong learning function here first, so I am keeping it to show how the learning function (among others) can vary quite a bit between different ART modules
        # self.W[j, :] = self.beta * x + (1-self.beta) * self.W[j, :]

        # The FuzzyART weight update rule is a linear interpolation between the old weight and the fuzzy intersection, and how far along that interpolation we go is set by beta (between [0,1]).
        self.W[j, :] = (1 - self.beta) * self.W[j, :] + self.beta * (torch.minimum(x, self.W[j, :]))

        # Because we wrote this function to update the weight in place, we have an empty return
        return

    # This function is the main training interface, taking one sample at a time
    def train(self, x: Tensor, y: int = -1):
        # First, we make sure that we have at least one category
        n_categories = self.W.shape[0]
        # If we don't have any categories, then immediately create one and update it
        if n_categories == 0:
            self.init_cat(x)
            if y != -1:
                self.map.append(y)
            return

        # Next, we compute the activations.
        # NOTE: There are two ways to do this:
        #   1. Preallocate a vector and iterate with a for loop
        #   2. List comprehension, where this is done all at once
        # List comprehension is sometimes slower for complicated low-level reasons, but it is a useful tool in lots of scenarios to make the syntax simpler.

        # OPTION 1: For loop way (commented out for illustration purposes)
        # T = torch.zeros(n_categories)
        # for j in range(n_categories):
        #     T[j] = self.activation(x, j)

        # OPTION 2: List comprehension way
        T = Tensor([self.activation(x, j) for j in range(n_categories)])

        # Next, we do the vigilance check.
        # This will involve going through the weights in order of highest activation and seeing if any of them pass the vigilance criterion.
        # For FuzzyART, this simply means if `Mj > rho`, where `rho` is the vigilance parameter.
        # The first one that passes this check wins and gets updated (hence winner-take-all).
        # If none do, then a new category is created and updated (hence the neurogenesis).

        # There are also two ways to handle the vigilance check programmatically:
        #   1. Sort the activations then iterate
        #   2. argmax the activations iteratively and zero them out if the don't pass.
        # Both methods have pros and cons, but we will go with the sorting procedure since it is simpler (even though it is technically slower since the sort is O(n long(n)) )

        # Sort the activations in order of highest activation first (in torch, this means descending order)
        # NOTE: we want the resulting indices too because we care about the original weight index that was highest activated
        T, inds = torch.sort(T, descending=True)

        # Create a flag that acts as the signal if any weight won
        did_match = False

        # Iterate over the activations in order of highest first
        for _, j in enumerate(T):
            # Extract the index of the current node with a bunch of type casting (ints can be used to index in Python, but torch Tensors can't)
            J = int(inds[int(j)])
            # Compute the match value at that index
            M = self.match(x, J)
            # If the match value is greater than the vigilance parameter, update that weight and stop the search
            if M > self.rho:
                # If we got a supervised label, break if the winning category has the wrong label
                if y != -1 and self.map[J] != y:
                    break
                # Update the weight according to the FuzzyART learning rule
                self.learn(x, J)
                # Raise the flag to say that we did have a winner
                did_match = True
                # Stop iterating over the weights
                break

        # If we didnt' have a winner, then create a new weight entirely and immediately update it (similar to how we do if we create the first weight at the top)
        if not did_match:
            self.init_cat(x)
            if y != -1:
                self.map.append(y)

        return

    # This function will classify a provided sample and report the index of the internal category that it belongs to.
    # NOTE: we also have a special option to get the "best-matching-unit" (bmu) in the case of complete mismatch (i.e., the sample was unrecognized), in which case we report the category that had the highest activation.
    def classify(self, x: Tensor, get_bmu: bool = True):
        # First, infer the number of categories that we currently have
        n_categories = self.W.shape[0]

        # Next compute the activations for each category using the list comprehension way
        T = Tensor([self.activation(x, j) for j in range(n_categories)])

        # Start out by saying that the reported value is mismatched, which we code with -1.
        # This will be hopefully overwritten if we have a match
        # NOTE: we initialize it out here because we want y_hat in this scope level so that it is correctly returned
        y_hat = -1

        # Sort the activations like before, keeping track of the corresponding indices
        T, inds = torch.sort(T, descending=True)

        # Have a match flag as before
        did_match = False
        # Iterate in order of highest activation
        for _, j in enumerate(T):
            # Extract the index of the corresponding weight
            J = int(inds[int(j)])
            # Compute the match function for the weight
            M = self.match(x, J)
            # If it satisfies the match criterion, report that category index as our winner
            if M > self.rho:
                # Set the output as that index and break out of the loop
                if self.map:
                    y_hat = self.map[J]
                else:
                    y_hat = J
                did_match = True
                break

        # If there was not a winner, we would should handle it appropriately
        if not did_match:
            # If we still want to report some actual value, return the highest activated category (which is first in the sorted list)
            if get_bmu:
                bmu = int(inds[0])
                if self.map:
                    y_hat = self.map[bmu]
                else:
                    y_hat = bmu
            # Otherwise, return a mismatch signal
            else:
                y_hat = -1

        return y_hat

# Here, we make a couple of helper functions to set up the plot


# Scatters points TSNE points and colors according to label
def add_2d_scatter(ax, points, colors, title=None):
    x, y = points
    ax.scatter(x, y, s=50, c=colors, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


# Generates the plot itself
def plot_2d(points, y, title):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in y]
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, colors)
    plt.show()


def get_tsne(data):
    # Now, we initialize a TSNE module with its own set of hyperparameters for how it will be "trained" to create a mapping between the 4d data and its 2d projection.
    t_sne = TSNE(
        n_components=2,
        perplexity=10,
        init="random",
        max_iter=250,
        random_state=0,
    )

    # The `x` points are the original columns, and the `y` points are FuzzyART's cluster labels of them
    x = data.data[["SL", "SW", "PL", "PW"]].values
    # y = y_hats

    # Fit the TSNE to the data, and return those transformed points
    S_t_sne = t_sne.fit_transform(x)
    return S_t_sne

#     # Generate the plot
# plot_2d(S_t_sne.T, y, "TSNE (FuzzyART Labels)")