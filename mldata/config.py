import logging
import os.path

from xdg.BaseDirectory import xdg_config_home
import yaml


logger = logging.getLogger(__name__)

_default_data_dir = os.path.join(xdg_config_home, 'mldata', 'data')
_config_path = os.path.join(xdg_config_home, 'mldata', 'config.yml')

config = {
    'root_dir': _default_data_dir
    }


def update(**kwargs):
    for key, value in kwargs.items():
        if key not in config:
            raise KeyError('Unexpected config key: {}'.format(key))
        config[key] = value


if os.path.exists(_config_path):
    with open(_config_path, 'r') as f:
        update(**yaml.load(f))

if config['root_dir'] == _default_data_dir:
    logging.warn("mldata data directory is still set to the default (%s). "
                 "Consider setting 'root_dir' in %s.",
                 _default_data_dir, _config_path)
