import linecache
import math
import pickle
import sqlite3
from pathlib import Path

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Embedding, Input, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import to_categorical
from tqdm import tqdm


def GeneratorPretraining(V, E, H):
    """Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V).
    """
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(None,), dtype="int32", name="Input")  # (B, T)
    out = Embedding(V, E, mask_zero=True, name="Embedding")(input)  # (B, T, E)
    # Build a chain of functions between Layers, Embedding layer to change the output shape
    out = LSTM(H, return_sequences=True, name="LSTM")(out)  # (B, T, H)
    # Constructing a chain of functions between Layers
    out = TimeDistributed(  # The TimeDistributed layer performs a Dense operation on each vector
        Dense(
            V,
            activation="softmax",
            name="DenseSoftmax",
        ),  # Define a neural layer with V nodes, using softmax activation function
        name="TimeDenseSoftmax",
    )(out)  # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining


class Generator:
    "Create Generator, which generate a next word."

    def __init__(self, sess, B, V, E, H, models_base_path, lr=1e-3):
        """# Arguments:
            B: int, Batch size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001.
        """
        self.sess = sess
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.models_base_path = models_base_path
        self.lr = lr
        self._build_gragh()
        self.reset_rnn_state()

    def _build_gragh(self):
        state_in = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 1),
        )  # Pass in the data, where None refers to the size of the batch size, which can be any number, and 1 refers to the size of the data
        h_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))  # (B,H)
        c_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))  # (B,H)
        action = tf.compat.v1.placeholder(tf.float32, shape=(None, self.V))  # onehot (B, V)
        reward = tf.compat.v1.placeholder(tf.float32, shape=(None,))  # (B, ) the rewards of each batch

        self.layers = []

        embedding = Embedding(
            self.V,
            self.E,
            mask_zero=True,
            name="Embedding",
        )  # The first layer is the Embedding layer
        out = embedding(state_in)  # Input is (B,V), output (B,1,E)
        self.layers.append(embedding)

        lstm = LSTM(self.H, return_state=True, name="LSTM")  # lstm layer
        out, next_h, next_c = lstm(out, initial_state=[h_in, c_in])  # Input is (B,1,E) and 2 (B,H), output is (B, H)
        self.layers.append(lstm)

        dense = Dense(self.V, activation="softmax", name="DenseSoftmax")  # Fully connected layer
        prob = dense(out)  # Input is (B,H), output is (B,V)
        self.layers.append(dense)

        log_prob = tf.math.log(tf.reduce_mean(input_tensor=prob * action, axis=-1))
        loss = -log_prob * reward
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        minimize = optimizer.minimize(loss)

        self.state_in = state_in
        self.h_in = h_in
        self.c_in = c_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.next_h = next_h
        self.next_c = next_c
        self.minimize = minimize
        self.loss = loss

        self.init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(self.init_op)

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):  # h、c: np.array, shape = (B,H)，ex.（32,64）
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True):
        """Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V).
        """
        feed_dict = {self.state_in: state, self.h_in: self.h, self.c_in: self.c}
        prob, next_h, next_c = self.sess.run(  # prob：np.array，shape=（B,V）ex.（32,1398）
            [self.prob, self.next_h, self.next_c],
            feed_dict,
        )

        if stateful:
            self.h = next_h
            self.c = next_c
            return prob
        else:
            return prob, next_h, next_c

    def update(self, state, action, reward, h=None, c=None, stateful=True):
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in: state,
            self.h_in: h,
            self.c_in: c,
            self.action: to_categorical(action, self.V),
            self.reward: reward,
        }
        _, loss, next_h, next_c = self.sess.run([self.minimize, self.loss, self.next_h, self.next_c], feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c

    def sampling_word(self, prob):
        action = np.zeros((self.B,), dtype=np.int32)
        for i in range(self.B):
            p = prob[i]  # p is an array of 1-dimensional V columns
            p /= p.sum()  # Total probability normalization
            action[i] = np.random.choice(
                self.V,
                p=p,
            )  # Probabilistically choose a number between 0 and V according to the probability provided by p

        return action

    def sampling_sentence(self, T, BOS=1):  # Generate sentences based on rnn
        self.reset_rnn_state()  # Parameters h,c state reset
        action = np.zeros([self.B, 1], dtype=np.int32)  # Generate a vertical array of size B, initially all 0
        action[:, 0] = BOS  # First full placement BOS
        actions = action
        # print(T)
        for _ in range(T):  # T is the maximum length of a sentence
            prob = self.predict(
                action,
            )  # Predicts the subsequent sequence based on the current sequence and returns the parameters
            action = self.sampling_word(prob).reshape(
                -1,
                1,
            )  # Sample words according to parameters and convert to 1 column
            for i in range(len(action)):
                if action[i][0] == 4:
                    action[i][0] = 3
            actions = np.concatenate(
                [actions, action],
                axis=-1,
            )  # The set of B sentences composed by the id of the word
        # Remove BOS
        actions = actions[
            :,
            1:,
        ]  # Take all the data from the first column to the right side of the data, i.e. all the data on the right side except column 0
        self.reset_rnn_state()

        return actions

    def generate_samples(self, T, g_data, num, output_file):
        print("Generate_samples was executed")
        sentences = []

        for _ in range(num // self.B + 1):
            actions = self.sampling_sentence(
                T,
            )  # Based on the neural network, the sentences are generated and returned as an array of the ids of the words

            actions_list = actions.tolist()  # Change from array to list

            for sentence_id in actions_list:
                # print("sentence_id",sentence_id)
                sentence = [
                    g_data.id2word[action] for action in sentence_id if action != 0 and action != 2
                ]  # Reverse the id to word, this is a concise version of the above
                sentences.append(sentence)  # Generators generate sentences
                # print(sentences)

        output_str = ""

        for i in range(num):
            output_str += " ".join(sentences[i]) + "\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_str)

        print("The generated sequence is written to", output_file)

    def sampling_rule(self, T, BOS=1):  # Generate rules based on rnn
        with open(Path(self.models_base_path / "id2word.txt")) as file:
            id2word = eval(file.read())

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)  # into a vertical array of size B, initially all 0
        actions = np.empty([self.B, 0], dtype=np.int32)
        action[:, 0] = BOS  # First full placement BOS

        for _ in range(T):  # T is the maximum length of a sentence
            prob = self.predict(
                action,
            )  # Predict the subsequent sequence based on the current sequence and return the parameters, which are input to the network in the shape of (B,1)

            action = self.sampling_word(prob).reshape(
                -1,
                1,
            )  # Sample words according to parameters and convert to 1 column
            np.argmax(prob, axis=-1).reshape([-1, 1])

            show2 = []
            for id in action:
                word = id2word[id[0]]
                show2.append(word)

            show3 = np.array(show2).reshape(-1, 1)
            actions = np.concatenate([actions, show3], axis=-1)  # Rule Growth

        self.reset_rnn_state()  # Reset the state of lstm, it is important to delete the equivalent of the above words to continue to predict, rather than based on the current state prediction

        return actions

    def generate_rules(self, T, g_data, num, output_file):
        print("Generate_rules was executed")
        # print(output_file)
        rules = []
        for _ in range(num // self.B + 1):
            actions = self.sampling_rule(T)  # Generate sentences based on neural networks
            print(actions)
            actions_list = actions.tolist()  # change from array to list, dimension remains the same, ex.(32,7)

            for rule_word in actions_list:  # Cycle B times
                rule = rule_word
                rules.append(rule)  # Generators generate sentences

        output_str = ""

        for i in range(num):
            # print(rules[i])
            for n in range(len(rules[i])):
                if rules[i][n] is None:
                    rules[i][n] = "<UNK>"
            # print(rules[i])
            output_str += ",".join(rules[i]) + "\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_str)

        print("The number of samples generated is", num)
        print("Written", output_file)

    def multipredict_rules_argmax(self, reason):  # Rules for generating many-to-one orders
        print("Executed multipredict_rules")

        with open(Path(self.models_base_path / "id2word.txt")) as file:
            id2word = eval(file.read())

        with open(Path(self.models_base_path / "word2id.txt")) as file:
            word2id = eval(file.read())

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)

        for i in range(len(reason)):
            word = reason[i]
            action[0][0] = word2id[word]
            prob = self.predict(action)
            result = np.argmax(prob, axis=-1).reshape([-1, 1])
            result = id2word[result[0][0]]

        return result

    def multipredict_rules_probability(self, reason):  # Rules for generating many-to-one orders
        print("Executed multipredict_rules")

        with open(Path(self.models_base_path / "id2word.txt")) as file:
            id2word = eval(file.read())

        with open(Path(self.models_base_path / "word2id.txt")) as file:
            word2id = eval(file.read())

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)

        for i in range(len(reason)):
            word = reason[i]
            try:
                action[0][0] = word2id[word]

            except:
                action[0][0] = "3"

            prob = self.predict(action)
            result = np.argmax(prob, axis=-1).reshape([-1, 1])
            result = id2word[result[0][0]]
            result_ = np.random.choice(self.V, p=prob[0])
            result_ = id2word[result_]

        return result_

    def train_rules(self, rule_len, path):
        print("Executed train_rules")
        print("Sequence sample size is", rule_len)

        with open(Path(self.models_base_path / "id2word.txt")) as file:
            id2word = eval(file.read())

        with open(Path(self.models_base_path / "word2id.txt")) as file:
            word2id = eval(file.read())

        rules_final = {}

        for idx in tqdm(range(rule_len)):
            sentence = linecache.getline(str(path), idx)  # Read the idx row of the original data
            words = sentence.strip().split(",")  # list type

            with open(Path(self.models_base_path / "att_name.txt")) as file:
                label2att = eval(file.read())

            # LHS multi-attribute function dependency, here if you want to restore the filtering at the time of rule generation, check this module of the E drive backup
            reason = words  # words: ['31301', 'BENSON', 'AZ', '85602', 'COCHISE']
            self.reset_rnn_state()  # Resetting the LSTM state
            action = np.zeros([self.B, 1], dtype=np.int32)

            for i in range(len(reason) - 2):  # len(reason)
                flag = i  # i is the last one of the current reason part, flag is the mark, is of course the first element in the reason location, when flag < 0, on behalf of the front has no use elements, that is, the reason part can not launch the result, give up the rule
                flag_ = (
                    flag
                )  # Add an index of information response, starting from the last position of REASON and moving forward

                while flag >= 0:
                    # print("flag",flag)
                    dic_name = []
                    left_ = []  # Used to store the reason part of the construction dictionary
                    word_ = []  # is used to store the name of the constructed dictionary, which is actually the same as dic_name
                    while flag_ != i:
                        # print("i=",i,"flag=",flag,"flag_=",flag_)
                        word = reason[flag_]
                        left = label2att[flag_]  # New Information
                        try:
                            action[0][0] = word2id[word]
                        except:
                            print("Not in the dictionary", word)
                            action[0][0] = "3"

                        # print("Add information",action,"即",left,":",word)    # Add explainable time display
                        prob = self.predict(action)
                        dic_name.append(word)
                        left_.append(left)
                        word_.append(word)
                        flag_ = flag_ + 1
                    word = reason[flag_]  # word: 31301;word: BENSON;word: AZ;word: 85602;word: COCHISE，One at a time
                    dic_name.append(word)
                    # print(word)
                    left = label2att[flag_]  # Get the attribute name of the corresponding position in the dictionary
                    right = label2att[i + 1]

                    try:
                        action[0][0] = word2id[word]
                    except:
                        action[0][0] = "3"
                        print("No such word in the dictionary")

                    prob = self.predict(action)
                    result = np.argmax(prob, axis=-1).reshape([-1, 1])  # Dictionary Index
                    result = id2word[result[0][0]]  # 实际内容

                    self.reset_rnn_state()

                    # Create dictionary for subsequent addition of partial rules, but sorting needs to be in front
                    for n in range(len(left_)):
                        if n == 0:
                            addtwodimdict(rules_final, str(dic_name), "reason", {str(left_[n]): str(word_[n])})
                        else:
                            addtwodimdict(rules_final[str(dic_name)], "reason", str(left_[n]), str(word_[n]))

                    if i == flag:
                        # print("At this point i=flag")
                        addtwodimdict(rules_final, str(dic_name), "reason", {str(left): str(word)})
                        addtwodimdict(rules_final, str(dic_name), "result", {str(right): str(result)})

                    else:
                        addtwodimdict(rules_final[str(dic_name)], "reason", str(left), str(word))
                        addtwodimdict(rules_final[str(dic_name)], "result", str(right), str(result))

                    # Save the rules you think are correct to the dictionary, delete the wrong rules
                    if result == reason[i + 1]:
                        # If it comes out, jump out of the loop, otherwise the flag is moved forward by one bit and additional information is added to continue the test
                        break
                    else:
                        # print("The predicted value does not match the actual value, increase the reason part")    # Add explainable time display
                        del rules_final[str(dic_name)]

                    flag = flag - 1
                    flag_ = flag

            with open(Path(self.models_base_path / "rules_final.txt"), "w") as file:
                file.write(str(rules_final))

        print("The rule generation is complete and the number of", len(rules_final))

    def filter(self, path):
        print("The filter is executed")

        with open(Path(self.models_base_path / "rules_final.txt")) as file:
            rules_final = eval(file.read())

        num = 0
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        for rulename, ruleinfo in list(rules_final.items()):
            num += 1

            left = list(ruleinfo["reason"].keys())
            word = list(ruleinfo["reason"].values())
            k = list(ruleinfo["result"].keys())
            right = k[0]
            v = list(ruleinfo["result"].values())
            result = v[0]

            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            while i < len(left):
                sqlex = sqlex + ' and "' + left[i] + "\"='" + word[i] + "'"
                i += 1

            sql1 = 'select "' + right + '" from "' + path + '" where "' + sqlex
            cursor.execute(sql1)
            rows = cursor.fetchall()
            num1 = len(rows)
            if num1 < 3:
                del rules_final[str(rulename)]
                continue
            else:
                t_rule = 1
                for row in rows:
                    if str(row[-1]) == str(
                        result,
                    ):  # In this case, the rule matches the data, and the confidence of the rule increases
                        t_rule = t_rule + 1
                        print("-->", t_rule, end="")
                    else:  # In this case, the rule is contrary to the data, and the confidence of the rule is reduced
                        t_rule = t_rule - 2
                        print("-->", t_rule, end="")
                rules_final[str(rulename)].update({"confidence": t_rule})  # Rule confidence initialization

        cursor.close()
        conn.close()

        with open(Path(self.models_base_path / "rules_final.txt"), "w") as file:
            file.write(str(rules_final))

        l2 = len(rules_final)
        print(
            "Rule filtering is complete and the remaining number of",
        )
        print(str(l2))

    def detect(self, rows, result, rulename, LHS, RHS, att2label, label2att):
        dert = 0
        t0 = 1
        t_rule = t0
        t_tuple = t0
        t_max = t_tuple  # The maximum value of confidence in different tuples satisfying the RULE condition
        flag = 1  # Flag whether the rule conflicts with the data
        flag_trust = 0  # 0 for believe data, 1 for believe rule
        for row in rows:
            if str(row[RHS]) == str(result):
                continue
            else:
                dert += 1
                flag = 0  # Mark the rule as conflicting with the data
        if flag == 1:  # If the rule does not conflict with the data, a great confidence level is given directly
            t_rule = t_rule + 100
            flag_trust = 3  # 3 means the rule is correct and there is no conflict
            return flag_trust

        else:  # The rule conflicts with the data, then the confidence of each tuple is calculated to adjust t_rule
            print("The rule conflicts with the data")
            print("Estimated changes for this restoration", dert)
            t_rule = t0
            for row in rows:  # Each tuple that satisfies the rule condition
                AV_p = []
                t_tp = 999  # The confidence of the current tuple is calculated as the minimum value of the confidence of different AV_i in a tuple, and a large value is set first to avoid the interference of the initial value.
                t_tc = t0

                for i in LHS:  # Calculate the minimum value of confidence in different AV_i in a tuple
                    AV_p.append(row[i])
                    t_AV_i = t0
                    attribute_p = label2att[i]

                    for rulename_p, ruleinfo_p in list(self.rule.items()):  # Traversing the dictionary
                        if rulename == rulename_p:
                            continue

                        if t_AV_i > 100 or t_AV_i < -100:
                            break

                        v = list(ruleinfo_p["result"].values())
                        left = list(ruleinfo_p["reason"].keys())
                        word = list(ruleinfo_p["reason"].values())
                        k = list(ruleinfo_p["result"].keys())
                        t_r = ruleinfo_p["confidence"]

                        if t_r < 0:
                            continue

                        right = k[0]
                        if attribute_p == right:
                            flag_equal = 0  # Can the rule determine the token of row[i]

                            for k in range(len(left)):
                                if (
                                    row[att2label[left[k]]] == word[k]
                                ):  # If the tuple where row[i] is located satisfies all AV_p of a rule, mark it as 1
                                    flag_equal = 1

                                else:
                                    flag_equal = 0
                                    break
                            if (
                                flag_equal == 1
                            ):  # If the row[i] in the tuple can be determined by other rules, check whether it satisfies the rule
                                result2 = v[0]

                                if (
                                    str(row[i]) == str(result2)
                                ):  # Retrieve other rules to determine the confidence level of each token in the tuple, increase if it is satisfied, and decrease if it is not.
                                    t_AV_i = t_AV_i + t_r
                                else:
                                    t_AV_i = t_AV_i - t_r
                                    print("The tuples that match the current rule are：", row)
                                    print(
                                        "In AV_p",
                                        str(row[i]),
                                        "with",
                                        str(result2),
                                        "does not match, the corresponding rule is",
                                        ruleinfo_p,
                                        "Its confidence level is",
                                        t_r,
                                    )

                    if t_tp > t_AV_i:
                        t_tp = t_AV_i

                for rulename_c, ruleinfo_c in list(self.rule.items()):  # Iterate through the dictionary, calculate t_c
                    if rulename == rulename_c:
                        continue
                    v = list(ruleinfo_c["result"].values())
                    left = list(ruleinfo_c["reason"].keys())
                    word = list(ruleinfo_c["reason"].values())
                    k = list(ruleinfo_c["result"].keys())
                    t_r = ruleinfo_c["confidence"]
                    if t_r < 0:
                        continue
                    right = k[0]
                    attribute_c = label2att[RHS]
                    if attribute_c == right:
                        flag_equal = 0  # Can the rule determine the token of row[i]
                        for k in range(len(left)):
                            if (
                                row[att2label[left[k]]] == word[k]
                            ):  # If the tuple in which AV_c is located satisfies all AV_p of a rule, mark it as 1
                                flag_equal = 1
                            else:
                                flag_equal = 0
                                break
                        if (
                            flag_equal == 1
                        ):  # If the AV_c in the tuple can be determined by other rules, check whether it satisfies the rules
                            result2 = v[0]
                            if str(row[RHS]) == str(result2):
                                t_tc = t_tc + t_r
                            else:
                                t_tc = t_tc - t_r
                                print("The tuples that match the current rule are：", row)
                                print(
                                    "In AV_c",
                                    str(row[RHS]),
                                    "with",
                                    str(result2),
                                    "does not match, the corresponding rule is",
                                    ruleinfo_c,
                                    "Its confidence level is",
                                    t_r,
                                )

                if t_tp == 999:  # means that all cells in it cannot be determined by other rules, reset its value to t0
                    t_tp = t0
                if t_tc < t_tp:
                    t_tuple = t_tc
                else:
                    t_tuple = t_tp

                if str(row[RHS]) == str(
                    result,
                ):  # The tuple data is consistent with the rule, and the confidence level increases
                    if t_tuple > 0:
                        t_rule = t_rule + math.ceil(math.log(1 + t_tuple))
                    else:
                        t_rule = t_rule + t_tuple
                    t_max = t_max
                    print("-->", t_rule, end="")
                else:  # If the tuple data violates the rule, calculate the confidence of the corresponding tuple
                    if t_tuple > 0:
                        t_rule = t_rule - 2 * t_tuple
                    else:
                        t_rule = t_rule + math.ceil(math.log(1 + abs(t_tuple)))
                        print("-->", t_rule, end="")

                    if t_rule < -100:
                        flag_trust = 0
                        return flag_trust  # In this case, the confidence level of the rule is too small, so the loop is directly jumped and marked as error.

                if t_max < t_tuple:
                    t_max = t_tuple

            print(
                "The final rule confidence level is",
                t_rule,
                "The tuple with which it conflicts has the highest confidence level of",
                t_max,
            )
        if t_rule > t_max:
            flag_trust = 1  # At this point the rule is considered correct and the data is modified
        elif t_rule < t_max:
            flag_trust = 0
            # print("The final rule confidence level is", t_rule, "The tuple with which it conflicts has the highest confidence level of", t_max)
            return flag_trust  # At this point the data is considered correct and the rule is modified
        self.rule[str(rulename)].update(
            {"confidence": t_rule},
        )  # Rule confidence initialization can be considered to be taken out separately
        print()
        return flag_trust

    def repair(self, iteration_num, path, order):
        print("Performed a REPAIR")

        with open(Path(self.models_base_path / "att_name.txt")) as file:
            label2att = eval(file.read())

        with open(Path(self.models_base_path / "rules_final.txt")) as file:
            self.rule = eval(file.read())

        att2label = {v: k for k, v in label2att.items()}  # Dictionary Reverse Transfer

        num = 0
        error_rule = 0
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        for rulename, ruleinfo in list(self.rule.items()):
            num += 1

            left = list(ruleinfo["reason"].keys())
            word = list(ruleinfo["reason"].values())
            k = list(ruleinfo["result"].keys())
            right = k[0]
            v = list(ruleinfo["result"].values())
            result = v[0]

            LHS = []
            LHS.append(att2label[left[0]])
            RHS = att2label[right]
            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            while i < len(left):
                sqlex = sqlex + ' and "' + left[i] + "\"='" + word[i] + "'"
                LHS.append(att2label[left[i]])
                i += 1

            cursor.execute('SELECT * FROM "' + path + '" where "' + sqlex)
            rows = cursor.fetchall()

            flag_trust = self.detect(rows, result, rulename, LHS, RHS, att2label, label2att)

            if (
                flag_trust == 3
            ):  # 3 means the rule is correct, and there is no conflict, proceed directly to the next rule
                continue

            if flag_trust == 0:
                error_rule += 1

            s1 = 0
            while flag_trust == 0 and s1 < 3:
                print("Rules can't be trusted, fix the rules")
                print("Repair rules right")
                s1 += 1
                result = self.multipredict_rules_probability(word)
                print("Right side changed to", result)
                flag_trust = self.detect(rows, result, rulename, LHS, RHS, att2label, label2att)

                if flag_trust == 1:
                    print("Rule repair successful")
                    addtwodimdict(self.rule, str(rulename), "result", {str(right): str(result)})
                    print("The modified rule is", self.rule[str(rulename)])

                elif flag_trust == 0 and s1 == 5:
                    print("No replacement fix on the right side of the rule")

            s2 = 0
            while flag_trust == 0 and s2 < 3:
                result = v[0]
                print("Repair rules left")
                s2 += 1
                min = 10
                flag = int(att2label[left[0]])

                if min > flag:
                    min = flag  # The index corresponding to the leftmost part of the current REASON section

                if min == 0:
                    print("No fixes can be added to the left side of the rule, delete the rule")
                    del self.rule[str(rulename)]
                    break
                left_new = label2att[min - 1]
                print("Add", left_new, "Information")
                sqladd = (
                    'select "'
                    + left_new
                    + '" from "'
                    + path
                    + '" where "'
                    + sqlex
                    + 'and "'
                    + right
                    + "\"='"
                    + result
                    + "'"
                )
                print("sqladd:", sqladd)
                cursor.execute(sqladd)
                rows_left = cursor.fetchall()

                # Reconstructing the dictionary
                if rows_left == []:
                    # print("There is no condition modified on the left side of the rule, delete the rule")
                    del self.rule[str(rulename)]
                    break
                addtwodimdict(self.rule[str(rulename)], "reason", str(left_new), str(rows_left[0][0]))
                for n in range(len(word)):
                    del self.rule[str(rulename)]["reason"][left[n]]
                    addtwodimdict(self.rule[str(rulename)], "reason", str(left[n]), str(word[n]))

                left = list(ruleinfo["reason"].keys())
                word = list(ruleinfo["reason"].values())

                sqlex = left[0] + "\"='" + word[0] + "'"
                i = 1
                while i < len(left):
                    sqlex = sqlex + ' and "' + left[i] + "\"='" + word[i] + "'"
                    i += 1
                sql1 = 'SELECT * FROM "' + path + '" where "' + sqlex
                cursor.execute(sql1)  # "City","State" ,where rownum<=10
                rows = cursor.fetchall()

                if len(rows) < 3:
                    continue

                result = self.multipredict_rules_argmax(word)
                # print(result)
                flag_trust = self.detect(rows, result, rulename, LHS, RHS, att2label, label2att)
                if flag_trust == 1:
                    print("Rule repair successful")
                    print("The modified rule is", self.rule[str(rulename)])
                elif flag_trust == 1 and min != 0:
                    del self.rule[str(rulename)]
                    break
            if flag_trust == 0:
                print("Rule is not available to fix, delete the rule")

            if flag_trust == 1:
                t0 = 1
                for row in rows:
                    if str(row[RHS]) == str(result):
                        continue
                    else:
                        AV_p = []
                        t_tp = 999  # The confidence of the current tuple is calculated as the minimum value of the confidence of different AV_i in a tuple, and a large value is set first to avoid the interference of the initial value.
                        t_tc = t0
                        flag_p = 0  # Used to record the position of the attribute corresponding to the lowest confidence level in AV_p
                        rule_p_name = []  # Record the rule with the highest confidence that can repair the attribute with the lowest confidence in the above AV_p
                        print("The tuples that match the current rule are: ", row)
                        for i in LHS:  # Calculate the minimum value of confidence in different AV_i in a tuple
                            AV_p.append(row[i])
                            t_AV_i = t0
                            attribute_p = label2att[i]
                            rulename_p_max = []
                            t_rmax = -999  # The maximum confidence level of the rules that correct AV_i in the following iterative dictionary, initially set to a minimal value
                            for rulename_p, ruleinfo_p in list(self.rule.items()):  # Traversing the dictionary
                                if rulename == rulename_p:
                                    continue
                                if t_AV_i > 100 or t_AV_i < -100:
                                    break
                                v = list(ruleinfo_p["result"].values())
                                left = list(ruleinfo_p["reason"].keys())
                                word = list(ruleinfo_p["reason"].values())
                                k = list(ruleinfo_p["result"].keys())
                                t_r = ruleinfo_p["confidence"]
                                if t_r < 0:
                                    continue
                                right = k[0]
                                if attribute_p == right:
                                    flag_equal = 0  # Can the rule determine the token of row[i]
                                    for k in range(len(left)):
                                        if (
                                            row[att2label[left[k]]] == word[k]
                                        ):  # If the tuple where row[i] is located satisfies all AV_p of a rule, mark it as 1
                                            flag_equal = 1
                                        else:
                                            flag_equal = 0
                                            break
                                    if (
                                        flag_equal == 1
                                    ):  # If the row[i] in the tuple can be determined by other rules, check whether it satisfies the rule
                                        result2 = v[0]
                                        if t_rmax < t_r:  # 记Record the maximum rule confidence in these rules
                                            t_rmax = t_rmax
                                            rulename_p_max = (
                                                rulename_p
                                            )  # Record the identification of the most trusted rule in the dictionary
                                        if str(row[i]) == str(result2):
                                            t_AV_i = t_AV_i + t_r
                                        else:
                                            t_AV_i = t_AV_i - t_r
                                            print(
                                                "In AV_p",
                                                str(row[i]),
                                                "with",
                                                str(result2),
                                                "does not match, the corresponding rule is",
                                                ruleinfo_p,
                                                "Its confidence level is",
                                                t_r,
                                            )

                            if t_tp > t_AV_i:
                                t_tp = t_AV_i
                                flag_p = i  # Record the index of AV_i with the lowest confidence
                                rule_p_name = (
                                    rulename_p_max
                                )  # Record the name of the rule that corrects this AV_i with the highest confidence

                        for rulename_c, ruleinfo_c in list(
                            self.rule.items(),
                        ):  # Iterate through the dictionary, calculate t_c
                            if rulename == rulename_c:
                                continue
                            v = list(ruleinfo_c["result"].values())
                            left = list(ruleinfo_c["reason"].keys())
                            word = list(ruleinfo_c["reason"].values())
                            k = list(ruleinfo_c["result"].keys())
                            t_r = ruleinfo_c["confidence"]
                            if t_r < 0:
                                continue
                            right = k[0]
                            attribute_c = label2att[RHS]
                            if attribute_c == right:
                                flag_equal = 0  # Can the rule determine the token of row[i]
                                for k in range(len(left)):
                                    if (
                                        row[att2label[left[k]]] == word[k]
                                    ):  # If the tuple in which AV_c is located satisfies all AV_p of a rule, mark it as 1
                                        flag_equal = 1
                                    else:
                                        flag_equal = 0
                                        break
                                if (
                                    flag_equal == 1
                                ):  # If the AV_c in the tuple can be determined by other rules, check whether it satisfies the rules
                                    result2 = v[0]
                                    if str(row[RHS]) == str(result2):
                                        t_tc = t_tc + t_r
                                    else:
                                        t_tc = t_tc - t_r
                                        print(
                                            "In AV_c",
                                            str(row[RHS]),
                                            "with",
                                            str(result2),
                                            "does not match, the corresponding rule is",
                                            ruleinfo_c,
                                            "Its confidence level is",
                                            t_r,
                                        )

                        if (
                            t_tp == 999
                        ):  # means that all cells in it cannot be determined by other rules, reset its value to t0
                            t_tp = t0
                        if t_tc < t_tp or t_tc == t_tp:
                            print(
                                "In this case, the data result is considered partially wrong, and the data is repaired according to the rule, the current rule is",
                                rulename,
                                "-->",
                                result,
                                "t_p is",
                                t_tp,
                                "t_c is",
                                t_tc,
                            )
                            for x in range(len(row) - 1):  # t2
                                if x == 0:
                                    sql_info = f"\"{label2att[x]}\"='{row[x]}'"
                                else:
                                    sql_info += f" and \"{label2att[x]}\"='{row[x]}'"
                            sql_update = (
                                'update "'
                                + path
                                + '" set "Label"=\'2\' , "'
                                + label2att[RHS]
                                + "\"='"
                                + result
                                + "' where  "
                                + sql_info
                                + ""
                            )
                            print("Original: ", sql_info)
                            print("Update Information: ", sql_update)
                            cursor.execute(sql_update)
                            conn.commit()
                        else:
                            print(rule_p_name)
                            if rule_p_name == []:
                                print("There may be errors")
                                continue
                            rname = self.rule[str(rule_p_name)]
                            v2 = list(rname["result"].values())
                            result2 = v2[0]
                            print(
                                "At this point, the data inference is considered partially wrong, and the data is "
                                "repaired according to the rule, the current rule is",
                                rule_p_name,
                                "-->",
                                result2,
                                "t_p is",
                                t_tp,
                                "t_c is",
                                t_tc,
                            )
                            for x in range(len(row) - 1):  # t2
                                if x == 0:
                                    sql_info = '"' + label2att[x] + "\"='" + row[x] + "'"
                                else:
                                    sql_info = sql_info + ' and "' + label2att[x] + "\"='" + row[x] + "'"
                            sql_update = (
                                'update "'
                                + path
                                + '" set "Label"=\'2\' , "'
                                + label2att[flag_p]
                                + "\"='"
                                + result2
                                + "' where  "
                                + sql_info
                                + ""
                            )
                            print("Original: ", sql_info)
                            print("Update Information: ", sql_update)
                            cursor.execute(sql_update)
                            conn.commit()
                            continue

        cursor.close()
        conn.close()

        print("Repair completed")
        print("Save repair rules")
        print("Rule dictionary size", len(self.rule))

        with open(Path(self.models_base_path / "rules_final.txt"), "w") as file:
            file.write(str(self.rule))

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)

        with open(path, "wb") as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, "rb") as f:
            weights = pickle.load(f)

        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)


def Discriminator(V, E, H=64, dropout=0.1):
    """Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1).
    """
    input = Input(shape=(None,), dtype="int32", name="Input")  # (B, T)
    out = Embedding(V, E, mask_zero=True, name="Embedding")(input)  # (B, T, E)
    out = LSTM(H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name="Dropout")(out)
    out = Dense(1, activation="sigmoid", name="FC")(out)

    discriminator = Model(input, out)
    return discriminator


def Highway(x, num_layers=1, activation="relu", name_prefix=""):
    """Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size).
    """
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = f"{name_prefix}Highway/Gate_ratio_{i}"
        fc_name = f"{name_prefix}Highway/FC_{i}"
        gate_name = f"{name_prefix}Highway/Gate_{i}"

        gate_ratio = Dense(input_size, activation="sigmoid", name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x


def addtwodimdict(thedict, key_a, key_b, val):
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a: {key_b: val}})
