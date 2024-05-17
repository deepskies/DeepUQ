DefaultsDE = {
    "common": {
        "out_dir": "./DeepUQResources/results/",
        "temp_config": "./DeepUQResources/temp/temp_config_DE.yml",
    },
    "data": {
        "data_path": "./data/",
        "data_engine": "DataLoader",
        "size_df": 1000,
        "noise_level": "low",
        "normalize": False,
        "val_proportion": 0.1,
        "randomseed": 42,
        "batchsize": 100,
        "generatedata": False,
    },
    "model": {
        "model_engine": "DE",
        "model_type": "DE",
        "loss_type": "bnll_loss",
        "n_models": 100,
        "init_lr": 0.001,
        "BETA": 0.5,
        "n_epochs": 100,
        "save_all_checkpoints": False,
        "save_final_checkpoint": False,
        "overwrite_final_checkpoint": False,
        "plot": False,
        "savefig": False,
        "verbose": False,
    },
    "plots_common": {
        "axis_spines": False,
        "tight_layout": True,
        "default_colorway": "viridis",
        "plot_style": "fast",
        "parameter_labels": ["$m$", "$b$"],
        "parameter_colors": ["#9C92A3", "#0F5257"],
        "line_style_cycle": ["-", "-."],
        "figure_size": [6, 6],
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
        "size_df": 1000,
        "noise_level": "low",
        "normalize": False,
        "val_proportion": 0.1,
        "randomseed": 42,
        "batchsize": 100,
        "generatedata": False,
    },
    "model": {
        # the engines are the classes, defined
        "model_engine": "DER",
        "model_type": "DER",
        "loss_type": "SDER",
        "init_lr": 0.001,
        "COEFF": 0.5,
        "n_epochs": 100,
        "save_all_checkpoints": False,
        "save_final_checkpoint": False,
        "overwrite_final_checkpoint": False,
        "plot": False,
        "savefig": False,
        "save_chk_random_seed_init": False,
        "rs": 42,
        "verbose": False,
    },
    "plots_common": {
        "axis_spines": False,
        "tight_layout": True,
        "default_colorway": "viridis",
        "plot_style": "fast",
        "parameter_labels": ["$m$", "$b$"],
        "parameter_colors": ["#9C92A3", "#0F5257"],
        "line_style_cycle": ["-", "-."],
        "figure_size": [6, 6],
    },
}
DefaultsAnalysis = {
    "common": {
        "dir": "./DeepUQResources/",
        "temp_config": "./DeepUQResources/temp/temp_config_analysis.yml",
    },
    "model": {
        "model_engine": "DE",
        "model_type": "DE",
        "n_models": 100,
        "n_epochs": 100,
        "BETA": 0.5,
        "COEFF": 0.5,
        "loss_type": "SDER"
    },
    "analysis": {
        "noise_level_list": ["low", "medium", "high"],
        "model_names_list": ["DER_wst", "DE_desiderata_2"],
        # ["DER_desiderata_2", "DE_desiderata_2"]
        "plot": True,
        "savefig": False,
        "verbose": False,
    },
    "plots": {"color_list":
              ["#F4D58D", "#339989", "#292F36", "#04A777", "#DF928E"]},
    # Pinks ["#EC4067", "#A01A7D", "#311847"]},
    # Blues: ["#8EA8C3", "#406E8E", "#23395B"]},
    "metrics_common": {
        "use_progress_bar": False,
        "samples_per_inference": 1000,
        "percentiles": [75, 85, 95],
    },
    "metrics": {
        "AllSBC": {},
        "CoverageFraction": {},
    },
}
