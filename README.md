# mldata

A framework to layer a standard interface over datasets commonly used in
machine learning.

## Goals

* Store datasets on disk in their canonical format, for interoperability
* Transparently download datasets on first use
* Handle common dataset tasks:
  * Splitting datasets
  * Data preprocessing
  * Minibatching
