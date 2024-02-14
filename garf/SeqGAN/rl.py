import numpy as np

from SeqGAN.models import Generator


class Agent:
    """On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    """

    def __init__(self, sess, B, V, E, H, models_base_path, lr=1e-3):
        """# Arguments:
            sess: tf.Session
            B: int, batch_size        #Number of samples selected for one training session
            V: int, Vocabrary size      #Number of all selectable actions
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001.
        """
        self.sess = sess
        self.num_actions = V
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.eps = 0.1  # Random selection probability
        self.models_base_path = models_base_path
        self.generator = Generator(
            sess=self.sess,
            B=self.B,
            V=self.V,
            E=self.E,
            H=self.H,
            lr=self.lr,
            models_base_path=self.models_base_path,
        )

    def act(self, state, epsilon=0, deterministic=False):
        """# Arguments:
            state: numpy array, dtype=int, shape = (B, t)
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1).
        """
        word = state[
            :,
            -1,
        ].reshape(
            [-1, 1],
        )  # [:,-1] takes the last column of data (the shape becomes a row); .reshape([-1, 1]) converts it to a column vector
        return self._act_on_word(word, epsilon=epsilon, deterministic=deterministic)

    def _act_on_word(self, word, epsilon=0, deterministic=False, PAD=0, EOS=2):
        """# Arguments:
            word: numpy array, dtype=int, shape = (B, 1),
                word indicates current word.
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1).
        """
        action = None
        is_PAD = word == PAD  # ex.[[false][true][false][false]]
        is_EOS = word == EOS  # ex.[[false][false][true][false]]
        is_end = is_PAD.astype(int) + is_EOS.astype(int)  # 0+0 when neither pad nor eos ,ex.[[0][1][1][0]]
        is_end = 1 - is_end  # At this point 1 means not finished, and vice versa when pad or eos is 0
        is_end = is_end.reshape([self.B, 1])
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        elif not deterministic:
            probs = self.generator.predict(
                word,
            )  # (B, V), sentence B. The prediction gets the next word in each sentence, giving the probability of occurrence of each word in the dictionary
            action = self.generator.sampling_word(probs).reshape([self.B, 1])  # Selecting words by probability
        else:
            probs = self.generator.predict(word)
            action = np.argmax(probs, axis=-1).reshape(
                [self.B, 1],
            )  # In the two-dimensional case axis=-1 is the same as axis=1, and the most likely word is chosen
        return action * is_end

    def reset(self):
        self.generator.reset_rnn_state()

    def save(self, path):
        self.generator.save(path)

    def load(self, path):
        self.generator.load(path)


class Environment:
    """On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    """

    def __init__(self, discriminator, data_generator, g_beta, n_sample=16):
        """Environment class for Reinforced Learning
        # Arguments:
            discriminator: keras model
            data_generator: SeqGAN.models.GeneratorPretrainingGenerator
            g_beta: SeqGAN.rl.Agent, copy of Agent
                params of g_beta.generator should be updated with those of original
                generator on regular occasions.
        # Optional Arguments
            n_sample: int, default is 16, the number of Monte Carlo search sample.
        """
        self.data_generator = data_generator
        self.B = data_generator.B
        self.T = data_generator.T
        self.n_sample = n_sample
        self.BOS = data_generator.BOS
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.reset()

    def get_state(self):
        if self.t == 1:
            return self._state
        else:
            return self._state[
                :,
                1:,
            ]  # If the current state is not the initial state, remove the BOS from the first column

    def reset(self):
        self.t = 1
        self._state = np.zeros([self.B, 1], dtype=np.int32)  # Reset to row B, column 1 form
        self._state[:, 0] = self.BOS  # Set the value of the 0th element of all rows to BOS
        self.g_beta.reset()

    def step(self, action):
        """Step t -> t + 1 and returns a result of the Agent action.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1),
                state is Y_0:t-1, and action is y_t
        # Returns:
            next_state: numpy array, dtype=int, shape = (B, t)
            reward: numpy array, dtype=float, shape = (B, 1)
            is_episode_end: bool
            info: dict.
        """
        self.t = self.t + 1

        reward = self.Q(action, self.n_sample)
        is_episode_end = self.t > self.T

        self._append_state(action)
        next_state = self.get_state()
        info = None

        return [next_state, reward, is_episode_end, info]

    def render(self, head=1):
        for i in range(head):
            # print(self.get_state())
            ids = self.get_state()[i]
            words = [self.data_generator.id2word[id] for id in ids.tolist()]
            # You can comment out everything from here on down
            for i in range(len(words)):
                if words[i] is None:
                    words[i] = "<UNK>"
            word_str = ""
            for word in words:
                word_str += f"{word}"
            print(word_str)
        print("-" * 80)

    def Q(self, action, n_sample=16):
        """State-Action value function using Rollout policy
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1).

        # Optional Arguments:
            n_sample: int, default is 16, number of samples for Monte Calro Search

        # Returns:
            reward: numpy array, dtype=float, shape = (B, ), State-Action value

        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        """
        h, c = self.g_beta.generator.get_rnn_state()
        reward = np.zeros([self.B, 1])
        if self.t == 2:  # step t
            Y_base = self._state  # Initial case
        else:
            Y_base = self.get_state()  # (B, t-1)

        if self.t >= self.T + 1:
            Y = self._append_state(action, state=Y_base)
            return self.discriminator.predict(Y)

        # Rollout
        for _ in range(n_sample):
            Y = Y_base
            self.g_beta.generator.set_rnn_state(h, c)
            y_t = self.g_beta.act(Y, epsilon=self.g_beta.eps)
            Y = self._append_state(y_t, state=Y)
            for _ in range(self.t + 1, self.T):
                y_tau = self.g_beta.act(Y, epsilon=self.g_beta.eps)
                Y = self._append_state(y_tau, state=Y)
            reward += self.discriminator.predict(Y) / n_sample

        return reward

    def _append_state(self, word, state=None):
        """# Arguments:
        word: numpy array, dtype=int, shape = (B, 1).
        """
        word = word.reshape(-1, 1)
        if state is None:
            self._state = np.concatenate([self._state, word], axis=-1)
        else:
            return np.concatenate([state, word], axis=-1)
