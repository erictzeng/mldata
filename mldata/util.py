import hashlib
import logging
import os.path

import requests

from mldata.config import config


logger = logging.getLogger(__name__)


def get_path(*args):
    """Return a path under the mldata data directory."""
    return os.path.join(config['root_dir'], *args)


def maybe_download(url, dest, sha1=None):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)
    if sha1 is not None:
        actual_sha1 = sha1_hash(dest)
        if actual_sha1 != sha1:
            raise RuntimeError('File {} mismatch (expected {}, got {})'
                               .format(dest, sha1, actual_sha1))
        else:
            logger.info('Verified integrity of %s', dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'w') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def sha1_hash(path, block_size=1024):
    """Return a string containing the hex SHA1 hash of the given file."""
    hasher = hashlib.sha1()
    with open(path, 'r') as f:
        block = f.read(block_size)
        while block:
            hasher.update(block)
            block = f.read(block_size)
    return hasher.hexdigest()
