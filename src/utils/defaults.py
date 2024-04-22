DefaultsDE = {
    "common":{
        "out_dir":"./DeepUQResources/results/",
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
        "wd": "./",
        "BETA": 0.5,
        "n_epochs": 100,
        "save_all_checkpoints": False,
        "save_final_checkpoint": True,
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
        "parameter_labels" : ['$m$','$b$'], 
        "parameter_colors": ['#9C92A3','#0F5257'], 
        "line_style_cycle": ["-", "-."],
        "figure_size": [6, 6]
    }, 
    "plots":{
        "CDFRanks":{}, 
        "Ranks":{"num_bins":None}, 
        "CoverageFraction":{}
    }, 
    "metrics_common": {
        "use_progress_bar": False,
        "samples_per_inference":1000, 
        "percentiles":[75, 85, 95]
    },
    "metrics":{
        "AllSBC":{}, 
        "CoverageFraction": {}, 
    }
}
DefaultsDER = {
    "common": {
        "out_dir": "./DeepUQResources/results/",
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
        "wd": "./",
        "COEFF": 0.5,
        "n_epochs": 100,
        "save_all_checkpoints": False,
        "save_final_checkpoint": True,
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
        "parameter_labels" : ['$m$','$b$'], 
        "parameter_colors": ['#9C92A3','#0F5257'], 
        "line_style_cycle": ["-", "-."],
        "figure_size": [6, 6]
    }, 
    "plots":{
        "CDFRanks":{}, 
        "Ranks":{"num_bins":None}, 
        "CoverageFraction":{}
    }, 
    "metrics_common": {
        "use_progress_bar": False,
        "samples_per_inference":1000, 
        "percentiles":[75, 85, 95]
    },
    "metrics":{
        "AllSBC":{}, 
        "CoverageFraction": {}, 
    }
}

