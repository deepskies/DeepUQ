DefaultsDE = {
    "common": {
        "out_dir": "./DeepUQResources/",
        "temp_config": "./DeepUQResources/temp/temp_config_DE.yml",
    },
    "data": {
        "data_path": "./data/",
        "data_engine": "DataLoader",
        "data_dimension": "2D",
        "data_injection": "input",
        "size_df": 1000,
        "noise_level": "low",
        "normalize": False,
        "uniform": False,
        "val_proportion": 0.1,
        "randomseed": 42,
        "batchsize": 100,
        "generatedata": False,
    },
    "model": {
        "model_engine": "DE",
        "model_type": "DE",
        "loss_type": "bnll_loss",
        "n_models": 5,
        "init_lr": 0.001,
        "BETA": 0.5,
        "n_epochs": 100,
        "save_all_checkpoints": False,
        "save_final_checkpoint": False,
        "overwrite_model": False,
        "plot_inline": False,
        "plot_savefig": False,
        "save_chk_random_seed_init": False,
        "rs_list": [41, 42],
        "save_n_hidden": False,
        "n_hidden": 64,
        "save_data_size": False,
        "verbose": False,
    },
}
DefaultsDER = {
    "common": {
        "out_dir": "./DeepUQResources/",
        "temp_config": "./DeepUQResources/temp/temp_config_DER.yml",
    },
    "data": {
        "data_path": "./data/",
        "data_engine": "DataLoader",
        "data_dimension": "2D",
        "data_injection": "output",
        "size_df": 1000,
        "noise_level": "low",
        "normalize": False,
        "uniform": False,
        "val_proportion": 0.1,
        "randomseed": 42,
        "batchsize": 100,
        "generatedata": False,
    },
    "model": {
        # the engines are the classes, defined
        "model_engine": "DER",
        "model_type": "DER",
        "loss_type": "DER",
        "init_lr": 0.001,
        "COEFF": 0.01,
        "n_epochs": 100,
        "save_all_checkpoints": False,
        "save_final_checkpoint": False,
        "overwrite_model": False,
        "plot_inline": False,
        "plot_savefig": False,
        "save_chk_random_seed_init": False,
        "rs": 42,
        "save_n_hidden": False,
        "n_hidden": 64,
        "save_data_size": False,
        "verbose": False,
    },
}
