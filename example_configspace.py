__author__ = 'gchlebus'


def get_configspace():
  import ConfigSpace as cs

  config = cs.ConfigurationSpace()
  config.add_hyperparameter(cs.CategoricalHyperparameter('dropout_rate', [0, 0.2, 0.4, 0.6, 0.8]))
  config.add_hyperparameter(cs.CategoricalHyperparameter('batch_normalization', [True, False]))
  config.add_hyperparameter(cs.CategoricalHyperparameter('upsampling_mode', ["transpose_conv", "upsampling"]))
  config.add_hyperparameter(cs.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-3, log=True))
  return config
