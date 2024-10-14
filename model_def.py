import tensorflow as tf


class Classifier(tf.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layers=1):
        self.layers = []
        for i in range(hidden_layers - 1):
            self.layers.append(Linear(num_inputs, num_inputs))
        self.layers.append(Linear(num_inputs, num_outputs))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = tf.nn.relu(x)
        return x


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))
        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )
        self.bias = bias
        if self.bias:
            self.b = tf.Variable(
                tf.zeros(shape=[1, num_outputs]),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w
        if self.bias:
            z += self.b
        return z
