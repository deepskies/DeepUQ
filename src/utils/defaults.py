Defaults = {
    "common":{
        "out_dir":"./DeepDiagnosticsResources/results/", 
        "temp_config": "./DeepDiagnosticsResources/temp/temp_config.yml", 
        "sim_location": "DeepDiagnosticsResources_Simulators"
    },
    "data": {
        "data_path": "./data/",
        "data_engine": "DataLoader",
        "size_df": 1000,
        "noise_level": "low",
        "val_proportion": 0.1,
        "randomseed": 42,
        "batchsize": 100,
    },
    "model": {
        "model_path": "./models/",
        # the engines are the classes, defined
        "model_engine": "DE",
        "n_models": 100,
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
