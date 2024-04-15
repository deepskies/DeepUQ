
from typing import Optional
import os 
import yaml 

from utils.defaults import Defaults

def get_item(section, item, raise_exception=True): 
    return Config().get_item(section, item, raise_exception) 

def get_section(section, raise_exception=True): 
    return Config().get_section(section, raise_exception)


class Config: 
    ENV_VAR_PATH = "DeepDiagnostics_Config"
    def __init__(self, config_path:Optional[str]=None) -> None:
        # okay what Maggie is doing here is a little trick or "cheat"
        # where the config_path is saved to the ENV_VAR_PATH
        # the first time this thing is called and then later it
        # can be loaded from this temp location saving on memory
        if config_path is not None: 
            # Add it to the env vars in case we need to get it later. 
            os.environ[self.ENV_VAR_PATH] = config_path
        else: 
            # Get it from the env vars 
            try: 
                config_path =  os.environ[self.ENV_VAR_PATH]
            except KeyError: 
                assert False, "Cannot load config from enviroment. Hint: Have you set the config path by pasing a str path to Config?"
        
        self.config = self._read_config(config_path)
        self._validate_config()

    def _validate_config(self): 
        # Validate common 
        # TODO 
        pass 

    def _read_config(self, path): 
        assert os.path.exists(path), f"Config path at {path} does not exist."
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    # if raise_exception is True, then throws an error if we're missing
    # otherwise, pull value from the defaults.py
    def get_item(self, section, item, raise_exception=True): 
        try: 
            return self.config[section][item]
        except KeyError as e: 
            if raise_exception: 
                raise KeyError(f"Configuration File missing parameter {e}")
            else: 
                return Defaults[section][item]

    def get_section(self, section, raise_exception=True): 
        try: 
            return self.config[section]
        except KeyError as e: 
            if raise_exception: 
                raise KeyError(e)
            else: 
                return Defaults[section]
