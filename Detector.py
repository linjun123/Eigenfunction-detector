import numpy as np
from jax import grad,jit,vmap
import jax.numpy as jnp
import jax.random as random
from jax.ops import index_update, index


from scipy.sparse.linalg import eigs,eigsh
from functools import partial
from scipy.sparse.linalg import LinearOperator, dsolve, spsolve
from scipy import linalg
from sklearn.preprocessing import scale
from scipy import sparse

import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.image import flip_left_right, flip_up_down

import time
from tqdm import trange # short cut for tqdm(range())
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns



@jit
def Hamilton(V, X):
    """
    Calculate the Hamiltonian as an operator

    Arg:
        V: potential
        X: a square matrix
    Return:
        The image of X under the Hamiltonian operator (-Laplacian + V)
    """
    X0 = (4 + V) * X
    X0 = index_update(X0, index[1:, :], X0[1:, :] - X[:-1, :])
    X0 = index_update(X0, index[:, 1:], X0[:, 1:] - X[:, :-1])
    X0 = index_update(X0, index[:-1, :], X0[:-1, :] - X[1:, :])
    X0 = index_update(X0, index[:, :-1], X0[:, :-1] - X[:, 1:])
    return X0


class Hamilton_operator(LinearOperator):
    def __init__(self, n, V, dtype=np.float32):
        self.V = V
        self.shape = (n * n, n * n)
        self.dtype = np.dtype(dtype)
        self.n = n

    def _matvec(self, x):
        return Hamilton(self.V, x.reshape(self.n, self.n)).ravel()

    def _rmatvec(self, x):
        return Hamilton(self.V, x.reshape(self.n, self.n)).ravel()


def mean_variance(n, p):
    """
    calculate the mean and distance variance of a 2d distribution p
    mean = Expectation_{p}((x,y))
    variance = Var_{p}(|(x,y)-mean|^{2})

    Arg:
        p: a square numpy matrix which is a probability distribution.
        n: the edge length of the matrix.

    Return:
        The mean and variance of distribution p.
    """
    xc = np.arange(0, n, 1)
    yc = np.arange(0, n, 1)
    xv, yv = np.meshgrid(xc, yc)
    mean = np.array([np.sum(xv * p), np.sum(yv * p)])
    variance = np.sum(((xv - mean[0]) ** 2) * p) + np.sum(((yv - mean[1]) ** 2) * p)
    return mean, variance


class Localization_landscape_detector:
    """
    The localization detector.

    Arg:
          key: a random key used to generate the samples
          n: the edge length of the square
          V_bar: the upper bound of the uniform distribution
          n_eivs: the number of the smallest eigenvalues and correponding eigenfunctions to calculate
    """

    def __init__(self, key, n=45, V_bar=3.0, n_eivs=1):

        self.n = n
        self.key = key
        self.n_samples = 0
        self.V_bar = V_bar
        self.training_labels = None
        self.training_landscapes = None
        self.effective_potentials = None
        self.samples = []
        self.model = None
        self.history = None
        self.ratio = None
        self.n_eivs = n_eivs

    def potential_generate(self, n_samples):
        """
        Generating random uniform potentials in [0,V_bar]
        For each potential, also generate the landscape and store it in self.training_landscapes
        For each potential, also generate the first n_eivs eigenvalues, eigenfunctions' locations and store them in self.training_labels

        Arg:
            n_samples: number of potentials to generate
        """
        print('generating potential samples...')
        self.n_samples += n_samples
        self.samples = list(self.samples)
        for i in trange(self.n_samples):
            self.key, _ = random.split(self.key)
            self.samples.append(random.uniform(self.key, (self.n, self.n), minval=0.0, maxval=self.V_bar))
        self.samples = jnp.array(self.samples)
        print('generating labels and landscapes...')
        labels_sci = []
        landscapes = []
        rhs = np.ones((self.n) ** 2, dtype=np.float32)
        for i in trange(self.n_samples):
            # print('here')
            Hamiltonian = Hamilton_operator(n=self.n, V=self.samples[i], dtype=np.float32)

            X2, exit_status = sparse.linalg.cg(Hamiltonian, rhs)[0].reshape((self.n, self.n)), \
                              sparse.linalg.cg(Hamiltonian, rhs)[1]
            if exit_status != 0:  # the convergence does not exit succesfully
                print('warning: convergence not succeed at sample {}\n'.format(i))
            landscapes.append(X2)

            w, v = eigsh(Hamiltonian, k=self.n_eivs, which='SM', maxiter=1e3)

            for j in range(self.n_eivs):
                v2 = v[:, j].reshape(self.n, self.n)
                mean, variance = mean_variance(self.n, (v2 ** 2) / np.sum(v2 ** 2))
                labels_sci.append(w[j] * self.n / np.log(self.n))
                labels_sci += [np.sqrt(variance) * self.n / np.sqrt(np.log(self.n)), mean[0], mean[1]]

        labels_sci = np.array(labels_sci)
        self.training_labels = labels_sci.reshape(self.n_samples, 4 * self.n_eivs)
        self.training_landscapes = np.expand_dims(np.array(landscapes), -1)

    def generate_effective_potentials(self):
        """
        Generating the effective potential which equals the reciprocal of landscape function
        """
        self.effective_potentials = np.reciprocal(self.training_landscapes)

    @partial(jit, static_argnums=(0,))
    def gradient(self, V, X):
        return (V + 4) * (Hamilton(V, X) - 1)

    def landscape_function_loss(self, V, num_iter=100, eta=0.02, nesterov=False):
        """
        calculate the landscape function for potential V

        Args:
            V: n by n potential
            num_iter: only meaningful when nesterov is True. The number of iterations to use in the gradient descent step
            eta: only meaningful when nesterov is True. The learning rate to use in the gradient descent step
            nesterov: whether to use the nesterov's method, if False, then use the Conjugate Gradient iteration method

        Return:
            if nesterov is True, return the n by n landscape function; else, return the landscape and the iteration status
        """
        if nesterov:
            X0 = np.zeros((self.n, self.n))
            Y = np.zeros((self.n, self.n))
            gamma = 0.0
            for j in range(num_iter):  # nesterov acceleration
                Z = X0 - eta * self.gradient(V, X0)
                X0 = Z + gamma * (Z - Y)
                Y = Z
                gamma = j / (j + 3)
            return Z
        else:
            rhs = np.ones((self.n) ** 2, dtype=np.float32)
            Ham = Hamilton_operator(self.n, V)
            return (sparse.linalg.cg(Ham, rhs)[0].reshape((self.n, self.n)), sparse.linalg.cg(Ham, rhs)[1])

    def new_model(self,architecture):
        print('generate a new cnn model...')

        model = architecture(nb_classes=4 * self.n_eivs, input_shape=(self.n, self.n, 1),
                             layer1_params=(3, 128, 2), res_layer_params=(3, 32, 25), reg=0.0)

        self.model = model

    def train(self, ratio, architecture, epochs=100, verbose=1, batch_size=36, use_landscape=True):
        """
        Training the model

        Args:
            ratio: a number between 0,1 which is the portion of training data
            architecture: a keras model
            other parameters: parameters for training a keras model
        """
        if self.model == None:
            self.new_model(architecture)

        self.ratio = ratio

        model = self.model
        print('compling the model...')
        adam = tf.keras.optimizers.Adam()
        model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])

        def schedule(epoch, lr):
            if epoch < 30:
                return 5e-3
            if epoch < 60:
                return 5e-4
            return 5e-5

        scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

        print('train the model with ratio:(1-ratio) train-test rate...')
        if use_landscape:
            self.history = model.fit(self.training_landscapes[:int(ratio * self.n_samples)],
                                     self.training_labels[:int(ratio * self.n_samples)],
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     validation_data=(self.training_landscapes[int(ratio * self.n_samples):],
                                                      self.training_labels[int(ratio * self.n_samples):]),
                                     callbacks=[scheduler])
        else:
            self.history = model.fit(self.effective_potentials[:int(ratio * self.n_samples)],
                                     self.training_labels[:int(ratio * self.n_samples)],
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     validation_data=(self.effective_potentials[int(ratio * self.n_samples):],
                                                      self.training_labels[int(ratio * self.n_samples):]),
                                     callbacks=[scheduler])
        self.model = model

    def plot_loss(self):
        """
        ploting the training process
        """
        plt.figure()
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def plot_test(self, n_egs, key):
        """
        plot some test examples

        Args:
            n_egs: number of examples
            key: random seed used to generate the n_egs examples
        """
        if self.ratio == None:
            self.ratio = 0.80
        test_indics = random.choice(key, jnp.array(range(int(self.ratio * self.n_samples), self.n_samples)), (n_egs,))
        predict_coor = self.model.predict(self.training_landscapes[test_indics, :])
        eiv_order = [i for i in range(1, self.n_eivs + 1)]

        for j in range(len(test_indics)):
            X2 = self.training_landscapes[test_indics[j], :][:, :, 0]
            plt.figure(figsize=(5, 5))
            plt.imshow(X2 / np.max(X2), cmap='jet', aspect='auto')
            plt.colorbar()
            for i in range(self.n_eivs):
                radius = self.training_labels[test_indics[j], 1 + 4 * i] * np.sqrt(np.log(self.n)) / self.n
                plt.text(self.training_labels[test_indics[j], 2 + 4 * i],
                         self.training_labels[test_indics[j], 3 + 4 * i], str(eiv_order[i]), c='black', fontsize=24)
                plt.scatter(self.training_labels[test_indics[j], 2 + 4 * i],
                            self.training_labels[test_indics[j], 3 + 4 * i],
                            s=490 + 360 * radius ** 2,
                            color="none",
                            edgecolor="black",
                            marker='s')

                radius_pred = predict_coor[j][1 + 4 * i] * np.sqrt(np.log(self.n)) / self.n
                plt.text(predict_coor[j][2 + 4 * i], predict_coor[j][3 + 4 * i], str(eiv_order[i]), c='white',
                         fontsize=24)
                plt.scatter(predict_coor[j][2 + 4 * i],
                            predict_coor[j][3 + 4 * i],
                            s=490 + 360 * radius_pred ** 2,
                            color="none",
                            edgecolor="white",
                            marker='s')
            plt.show()
        return predict_coor, self.training_labels[test_indics, :]

    def predict(self, V):
        land = np.expand_dims(self.landscape_function_loss(V)[0], -1)
        return self.model.predict(np.array([land]))

