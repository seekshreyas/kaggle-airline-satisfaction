#! /bin/bash

# repository tasks
pixi run sync-agents

# download dataset

kaggle competitions download -c playground-series-s6e10 -p data/raw
unzip data/raw/playground-series-s6e10.zip


kaggle competitions submit -c playground-series-s6e10 -f submission.csv  -m "Message"
