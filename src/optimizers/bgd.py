import tensorflow as tf


class BGDOptimizer(tf.keras.optimizers.Optimizer):
    """Custom optimizer for batch gradient descent"""

    def __init__(self, lr=0.01, **kwargs):
        super(BGDOptimizer, self).__init__(name="BatchGradientDescent", **kwargs)
        self._learning_rate = self._build_learning_rate(lr)

    def build(self, var_list):
        super(BGDOptimizer, self).build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.accumulators = []
        for var in var_list:
            self.accumulators.append(
                self.add_variable_from_reference(var, "accumulator")
            )
        self._built = True

    def update_step(self, gradient, variable):
        lr = tf.cast(self._learning_rate, dtype=variable.dtype)
        # w = self.get_slot(variable, "accumulator")
        var_key = self._var_key(variable)
        w = self.accumulators[self._index_dict[var_key]]
        variable.assign_sub(lr * gradient + w)

    def get_config(self):
        base_config = super(BGDOptimizer, self).get_config()
        base_config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
            }
        )
        return base_config
