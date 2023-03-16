def set_config(c):
    c.data_path = "data/example_CMS/example_CMS_data.npy"
    c.names_path = "data/example_CMS/example_CMS_names.npy"
    c.energy_conversion = False
    c.compression_ratio = 2.0
    c.epochs = 10
    c.early_stopping = False
    c.lr_scheduler = False
    c.patience = 100
    c.min_delta = 0
    c.model_name = "george_SAE"
    c.custom_norm = False
    c.l1 = True
    c.reg_param = 0.001
    c.RHO = 0.05
    c.lr = 0.001
    c.batch_size = 512
    c.save_as_root = True
    c.test_size = 0.15
    c.energy_conversion = False
    c.type_list = [
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "float64",
        "float64",
        "float64",
        "int",
        "int",
    ]
