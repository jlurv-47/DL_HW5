import tensorflow as tf


class AdamW:  # implements AdamW optimizer as decscribed in Loshchilov and Hutter 2019
    def __init__(
        self, alpha=1e-3, beta_one=9e-1, beta_two=9.99e-1, epsilon=1e-8, lam=1e-4, nu=1
    ):
        self.alpha = alpha
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.epsilon = epsilon
        self.lam = lam
        self.nu = nu
        self.t = 0
        self.m = {}  # storing moment vectors as dicts keyed by tf.variable.ref()
        self.v = {}

    def grad_update(self, grads, vars):
        self.t += 1
        for var, grad in zip(vars, grads):
            var_ref = (
                var.ref()
            )  # var.ref() returns hashable reference to tf.variable var, usable as dict key
            if var_ref not in self.m:
                self.m[var_ref] = tf.zeros_like(var)
                self.v[var_ref] = tf.zeros_like(var)
            self.m[var_ref] = (
                self.beta_one * self.m[var_ref] + (1 - self.beta_one) * grad
            )
            self.v[var_ref] = self.beta_two * self.v[var_ref] + (
                1 - self.beta_two
            ) * tf.square(grad)
            m_hat = self.m[var_ref] / (1 - self.beta_one**self.t)
            v_hat = self.v[var_ref] / (1 - self.beta_two**self.t)
            var.assign_sub(
                self.nu
                * (
                    self.alpha * m_hat / (tf.math.sqrt(v_hat) + self.epsilon)
                    + self.lam * var
                )
            )

    def set_schedule_multiplier(self, multiplier):
        self.nu = multiplier
