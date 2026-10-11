#! /bin/bash

# repository tasks
pixi run sync-agents

# download dataset

kaggle competitions download -c playground-series-s6e10 -p data/raw
