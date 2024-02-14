from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

from SeqGAN.models import Discriminator, GeneratorPretraining
from SeqGAN.rl import Agent, Environment
from SeqGAN.utils import DiscriminatorGenerator, GeneratorPretrainingGenerator

sess = tf.compat.v1.Session()
import keras.backend as K

K.set_session(sess)


class Trainer:
    """Manage training."""

    def __init__(
        self,
        order,
        B,
        T,
        g_E,
        g_H,
        d_E,
        d_H,
        d_dropout,
        generate_samples,
        path_pos,
        path_neg,
        path_rules,
        models_base_path,
        g_lr=1e-3,
        d_lr=1e-3,
        n_sample=16,
        init_eps=0.1,
    ):
        self.B, self.T = B, T  # batch size，max_length
        self.order = order
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps  # Exploration rate ϵ. i.e., the strategy is to select the current maximum value action with probability 1-ϵ and to select the new action at random with probability ϵ
        self.models_base_path = models_base_path
        self.path_pos = path_pos  # Address where the original data is located
        self.path_neg = path_neg  # Address where data is generated
        self.path_rules = path_rules

        self.g_data = GeneratorPretrainingGenerator(
            self.path_pos,
            order=order,
            B=B,
            T=T,
            models_base_path=self.models_base_path,
            min_count=1,
        )  # next method produces x, y_true data; both are the same data, e.g. [BOS, 8, 10, 6, 3, EOS], [8, 10, 6, 3, EOS]
        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            order=order,
            path_neg=self.path_neg,
            B=self.B,
            shuffle=True,
        )  # The next method produces pos data and neg data

        self.V = self.g_data.V  # Total vocabulary in the data
        self.agent = Agent(sess=sess, B=B, V=self.V, E=g_E, H=g_H, lr=g_lr, models_base_path=self.models_base_path)
        self.g_beta = Agent(sess=sess, B=B, V=self.V, E=g_E, H=g_H, lr=g_lr, models_base_path=self.models_base_path)

        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)

        self.env = Environment(self.discriminator, self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(
            self.V,
            g_E,
            g_H,
        )  # A 4-layer neural network input-embedding-lstm-dense

        self.rule = {}

    def pre_train(
        self,
        g_epochs=3,
        d_epochs=1,
        g_pre_path=None,
        d_pre_path=None,
        g_lr=1e-3,
        d_lr=1e-3,
    ):  # The actual parameters are given by the config
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)

        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        print("Pre-training generator")
        if g_pre_path is None:
            self.g_pre_path = Path(self.models_base_path / "generator_pre.hdf5")
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(
            g_adam,
            "categorical_crossentropy",
        )  # Training is performed, the optimizer is Adam, and the loss function is a categorical cross-entropy function for multi-classification
        print("Generator pre-training")
        self.generator_pre.summary()  # model.summary() in keras is used to output the status of the parameters of each layer of the model
        # print("++++++++++++++++++++")
        self.generator_pre.fit_generator(  # The return value is a History object. Its History.history property is a record of the training loss and evaluation values for successive epochs, as well as the validation set loss and evaluation values
            self.g_data,  # This should be an instance of a generator or Sequence (keras.utils.Sequence) object
            steps_per_epoch=None,
            epochs=g_epochs,
        )
        self.generator_pre.save_weights(self.g_pre_path)  # Save the weights to generator_pre.hdf5
        self.reflect_pre_train()  # Mapping layer layer weights to agent

    def pre_train_discriminator(self, d_epochs=1, d_pre_path=None, lr=1e-3):
        print("Pre-training discriminator")
        if d_pre_path is None:
            self.d_pre_path = Path(self.models_base_path / "discriminator_pre.hdf5")
        else:
            self.d_pre_path = d_pre_path

        print("Start Generating sentences")
        self.agent.generator.generate_samples(
            self.T,
            self.g_data,
            self.generate_samples,
            self.path_neg,
        )  # The generator generates a sequence that writes the txt at the output_file location

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,  # Sampling of real data
            order=self.order,
            path_neg=self.path_neg,  # Read the data just generated by the generator
            B=self.B,
            shuffle=True,
        )

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, "binary_crossentropy")
        self.discriminator.summary()
        print("Discriminator pre-trainin")

        self.discriminator.fit_generator(self.d_data, steps_per_epoch=None, epochs=d_epochs)
        self.discriminator.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def reflect_pre_train(self):  # Mapping layer layer weights to agent
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:  # If the weight of this layer is not 0
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(
                    w,
                )  # then the weight of the corresponding layer in agent is set to w
                self.g_beta.generator.layers[i].set_weights(w)
                i += 1

    def train(
        self,
        steps=10,
        g_steps=1,
        d_steps=1,
        d_epochs=1,
        g_weights_path="generator.pkl",
        d_weights_path="discriminator.hdf5",
        verbose=True,
        head=1,
    ):
        print("Start of official training")
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, "binary_crossentropy")
        self.eps = self.init_eps
        for step in range(steps):
            print("Current overall number of rounds", step + 1)
            # Generator training
            for _ in range(g_steps):
                print("G-step")
                rewards = np.zeros([self.B, self.T])  # reward creates an empty table
                self.agent.reset()  # agent reset
                self.env.reset()  # env reset
                for t in range(self.T):  # Start iteratively training the generator
                    state = self.env.get_state()

                    action = self.agent.act(state, epsilon=0.0)

                    _next_state, reward, is_episode_end, _info = self.env.step(action)
                    self.agent.generator.update(state, action, reward)
                    rewards[:, t] = reward.reshape(
                        [
                            self.B,
                        ],
                    )
                    if is_episode_end:
                        if verbose:
                            print(f"Reward: {np.average(rewards):.3f}, Episode end")
                            self.env.render(head=head)
                        break

            # Discriminator training
            for _ in range(d_steps):
                print("D-step")
                self.agent.generator.generate_samples(self.T, self.g_data, self.generate_samples, self.path_neg)
                self.d_data = DiscriminatorGenerator(
                    path_pos=self.path_pos,
                    order=self.order,
                    path_neg=self.path_neg,
                    B=self.B,
                    shuffle=True,
                )
                self.discriminator.fit_generator(self.d_data, steps_per_epoch=None, epochs=d_epochs)

            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps * (1 - float(step) / steps * 4), 1e-4)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    def generate_rules(self, file_name, generate_samples):
        path_rules = Path(self.models_base_path / file_name)
        print(path_rules)

        self.agent.generator.generate_rules(8, self.g_data, generate_samples, path_rules)

    def train_rules(self, rule_len, path):
        path_rules = Path(self.models_base_path / path)
        self.agent.generator.train_rules(rule_len, path_rules)

    def filter(self, path):
        self.agent.generator.filter(path)

    def repair(self, path):
        self.agent.generator.repair(1, path, self.order)  # 3
