def set_config(c):
    c.input_path = "workspaces/public_datasets/data/exafel1.npz"
    c.compression_ratio = 100
    # c.number_of_columns = 24
    # c.latent_space_size = 15
    c.epochs = 1000
    c.early_stopping = False
    c.early_stopping_patience = 100
    c.min_delta = 0
    c.lr_scheduler = True
    c.lr_scheduler_patience = 50
    # c.model_name = "Conv_AE_GDN"
    # c.model_type = "convolutional"
    c.model_name = "CFD_dense_AE"
    # c.model_name = "AE_Dropout_BN"
    c.model_type = "dense"
    c.custom_norm = True
    c.l1 = True
    c.reg_param = 0.001
    c.RHO = 0.05
    c.lr = 0.001
    c.batch_size = 32
    c.test_size = 0.2
    c.data_dimension = 2
    c.apply_normalization = False
    c.extra_compression = False
    c.intermittent_model_saving = False
    c.intermittent_saving_patience = 100
    c.activation_extraction = False
    c.deterministic_algorithm = False
    c.compress_to_latent_space = False
    c.save_error_bounded_deltas = False
    c.error_bounded_requirement = False
    c.convert_to_blocks = [1, 25, 25]
    # c.custom_loss_function = "loss_function_swae"


# def set_config(c):
#     c.input_path = "workspaces/CFD_workspace/data/CFDAnimation.npz"
#     c.data_dimension = 2
#     c.compression_ratio = 2.0
#     c.apply_normalization = False
#     c.model_name = "CFD_dense_AE"
#     c.epochs = 2
#     c.lr = 0.001
#     c.batch_size = 1
#     c.early_stopping = True
#     c.lr_scheduler = False

#     # === Additional configuration options ===

#     c.early_stopping_patience = 100
#     c.min_delta = 0
#     c.lr_scheduler_patience = 50
#     c.custom_norm = True
#     c.l1 = True
#     c.reg_param = 0.001
#     c.RHO = 0.05
#     c.test_size = 0
#     c.extra_compression = False
#     c.intermittent_model_saving = False
#     c.intermittent_saving_patience = 100
#     c.mse_avg = False
#     c.mse_sum = True
#     c.emd = False
#     c.l1 = True
#     c.activation_extraction = False
#     c.deterministic_algorithm = False
