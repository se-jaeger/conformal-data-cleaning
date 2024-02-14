from conformal_data_cleaning.data import CorruptionType

error_fractions = [0.01, 0.05, 0.1, 0.3, 0.5]
error_types = [x.value for x in CorruptionType]
confidence_levels = [0.5, 0.8, 0.999]
method_hyperparameters = {"AutoGluon": confidence_levels, "PyodECOD": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.499]}
