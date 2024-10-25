from typing import Optional
import os
import yaml
from deepuq.utils.defaults import (
    DefaultsDE,
    DefaultsDER,
    DefaultsAnalysis,
)


def get_item(section, item, raise_exception=True):
    """Retrieve an item from the configuration using the provided section and
    key.

    Parameters:
    ----------
    section : str
        The section in the configuration file.
    item : str
        The key within the section to retrieve.
    raise_exception : bool, optional
        Whether to raise an exception if the item is missing (default is True).

    Returns:
    -------
    Any
        The value of the item from the configuration.
    """
    return Config().get_item(section, item, raise_exception)


def get_section(section, raise_exception=True):
    """Retrieve an entire section from the configuration.

    Parameters:
    ----------
    section : str
        The section in the configuration file to retrieve.
    raise_exception : bool, optional
        Whether to raise an exception if the section is missing
        (default is True).

    Returns:
    -------
    dict
        The section from the configuration as a dictionary.
    """
    return Config().get_section(section, raise_exception)


class Config:
    """A class to handle reading and managing a configuration file.

    Attributes:
    -----------
    ENV_VAR_PATH : str
        The environment variable that stores the configuration path.

    Methods:
    --------
    get_item(section, item, defaulttype, raise_exception=True):
        Retrieves a specific item from the configuration or default values.
    get_section(section, defaulttype, raise_exception=True):
        Retrieves an entire section from the configuration or default values.
    """

    ENV_VAR_PATH = "DeepUQ_Config"

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the Config object, reading the configuration from a
        specified path or from an environment variable if the path is not
        provided.

        Parameters:
        ----------
        config_path : Optional[str]
            Path to the configuration file. If not provided, it is retrieved
            from the environment variable.
        """
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
                config_path = os.environ[self.ENV_VAR_PATH]
            except KeyError:
                assert (
                    False
                ), "Cannot load config from enviroment. \
                     Hint: Have you set the config path \
                     by passing a str path to Config?"
        self.config = self._read_config(config_path)
        self._validate_config()

    def _validate_config(self):
        """
        Validate the contents of the loaded configuration. This method is a
        placeholder for future implementation.
        """
        # Validate common
        # TODO
        pass

    def _read_config(self, path):
        """
        Read the configuration file from the provided path.

        Parameters:
        ----------
        path : str
            Path to the configuration file.

        Returns:
        -------
        dict
            The configuration data loaded from the file.

        Raises:
        -------
        AssertionError
            If the provided config path does not exist.
        """
        assert os.path.exists(path), f"Config path at {path} does not exist."
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    # if raise_exception is True, then throws an error if we're missing
    # otherwise, pull value from the defaults.py
    def get_item(
        self,
        section,
        item,
        defaulttype,
        raise_exception=True,
    ):
        """Retrieve a specific item from the configuration. If the item is
        missing, it can either raise an exception or retrieve the value from
        the default configuration.

        Parameters:
        ----------
        section : str
            The section in the configuration file.
        item : str
            The key within the section to retrieve.
        defaulttype : str
            The type of defaults to use if the item is missing
            ('DER', 'DE', 'Analysis').
        raise_exception : bool, optional
            Whether to raise an exception if the item is missing
            (default is True).

        Returns:
        -------
        Any
            The value of the item from the configuration or default
            configuration.

        Raises:
        -------
        KeyError
            If the item is missing and raise_exception is True.
        """
        try:
            return self.config[section][item]
        except KeyError as e:
            if raise_exception:
                raise KeyError(f"Configuration File missing parameter {e}")
            else:
                return {
                    "DER": DefaultsDER,
                    "DE": DefaultsDE,
                    "Analysis": DefaultsAnalysis,
                }[defaulttype][section][item]

    def get_section(self, section, defaulttype, raise_exception=True):
        """Retrieve an entire section from the configuration. If the section
        is missing, it can either raise an exception or retrieve the section
        from the default configuration.

        Parameters:
        ----------
        section : str
            The section in the configuration file to retrieve.
        defaulttype : str
            The type of defaults to use if the section is missing
            ('DER', 'DE', 'Analysis').
        raise_exception : bool, optional
            Whether to raise an exception if the section is missing
            (default is True).

        Returns:
        -------
        dict
            The section from the configuration or default configuration.

        Raises:
        -------
        KeyError
            If the section is missing and raise_exception is True.
        """
        try:
            return self.config[section]
        except KeyError as e:
            if raise_exception:
                raise KeyError(e)
            else:
                return {
                    "DER": DefaultsDER,
                    "DE": DefaultsDE,
                    "Analysis": DefaultsAnalysis,
                }[defaulttype][section]
