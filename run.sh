#! /bin/bash
# Reusable project commands. Run one task at a time:
#
#   ./run.sh <task> [args]      e.g.  ./run.sh cv configs/experiments/lr_01_ratings_onehot.yaml
#   ./run.sh                    list all tasks
#
# Everything runs through Pixi. Kaggle commands read credentials from .env.

set -euo pipefail
cd "$(dirname "$0")"

COMPETITION="playground-series-s6e10"
DATA_DIR="data/raw"
COMP_CFG="configs/competition.yaml"
EXP_DIR="configs/experiments"
REPORT_CFG="configs/reports/phase2_feature_engineering.yaml"
NOTEBOOK="notebooks/bruce_v1.py"

load_env() {  # export variables from .env (e.g. KAGGLE_API_TOKEN) for the kaggle CLI
    if [[ -f .env ]]; then set -a; source .env; set +a; fi
}

usage() {
    cat <<EOF
Usage: ./run.sh <task> [args]

Setup
  setup                       install the Pixi environment
  sync-agents                 regenerate agent instruction adapters (CLAUDE.md, Copilot, Cursor)
  sync-check                  check whether the adapters are up to date
  download                    download and unzip the competition data into ${DATA_DIR}/
  notebook [file]             open a marimo notebook in watch mode (default ${NOTEBOOK})
  test                        run the test suite

Data
  profile                     rows, nulls and target rate per competition file
  columns [csv]               show how raw column names are renamed (default train.csv)
  splits [salt]               holdout / dev / fold sizes and target rates (optional salt override)
  config <yaml>               print a fully merged experiment config

Experiments (dev subset, k-fold CV)
  features                    list registered feature blocks
  plots                       list the plots in the src/viz catalog
  cv <yaml> [yaml ...]        run CV for one or more experiment configs
  ladder-all                  re-run the whole Phase 2 feature ladder (lr_00 .. lr_07)
  grid                        run the Phase 3 regularization grid (lr_08)
  ladder                      latest run per experiment vs its parent, with keep/revert verdicts
  leaderboard [top]           all runs, best CV first

Final model and submission
  train [yaml]                full-data CV + holdout (once) + test predictions (default lr_final_full)
  validate <csv>              check a submission file against the sample submission
  submit <csv> "<message>"    validate, confirm y/N, then upload to Kaggle
  submissions                 list your Kaggle submissions and public scores

Reports
  diagnose <experiment>       save all diagnostic charts for a run as PNGs in reports/plots/<run_id>/
  report [yaml]               build the executive Deckset deck + PNGs (default ${REPORT_CFG})
EOF
}

task="${1:-help}"
shift || true

case "$task" in
    # --- setup ---------------------------------------------------------------
    setup)        pixi install ;;
    sync-agents)  pixi run sync-agents ;;
    sync-check)   pixi run python scripts/sync_agent_config.py --check ;;
    download)
        load_env
        pixi run kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"
        unzip -o "$DATA_DIR/$COMPETITION.zip" -d "$DATA_DIR/$COMPETITION"
        ;;
    notebook)     pixi run marimo edit --watch "${1:-$NOTEBOOK}" ;;
    test)         pixi run test ;;

    # --- data ----------------------------------------------------------------
    profile)      pixi run python -m src.datahelpers.dataloader profile "$COMP_CFG" ;;
    columns)      pixi run python -m src.datahelpers.datawrangle columns "${1:-$DATA_DIR/$COMPETITION/train.csv}" ;;
    splits)
        if [[ $# -ge 1 ]]; then
            pixi run python -m src.datahelpers.splits summary "$COMP_CFG" --salt "$1"
        else
            pixi run python -m src.datahelpers.splits summary "$COMP_CFG"
        fi
        ;;
    config)       pixi run python -m src.utils.config show "${1:?usage: ./run.sh config <yaml>}" ;;

    # --- experiments -----------------------------------------------------------
    features)     pixi run python alfred.py features ;;
    plots)        pixi run python -c "from src.viz import catalog; print(catalog().to_string(index=False))" ;;
    cv)
        [[ $# -ge 1 ]] || { echo "usage: ./run.sh cv <yaml> [yaml ...]" >&2; exit 1; }
        pixi run cv "$@"
        ;;
    ladder-all)
        pixi run cv "$EXP_DIR"/lr_00_baseline.yaml "$EXP_DIR"/lr_01_ratings_onehot.yaml \
            "$EXP_DIR"/lr_02_rating_flags.yaml "$EXP_DIR"/lr_03_delays.yaml "$EXP_DIR"/lr_04_splines.yaml \
            "$EXP_DIR"/lr_05_crosses.yaml "$EXP_DIR"/lr_06_rating_aggregates.yaml "$EXP_DIR"/lr_07_rating_crosses.yaml
        ;;
    grid)         pixi run cv "$EXP_DIR/lr_08_regularization.yaml" ;;
    ladder)       pixi run ladder ;;
    leaderboard)  pixi run python alfred.py leaderboard --top "${1:-20}" ;;

    # --- final model and submission ----------------------------------------------
    train)        pixi run train "${1:-$EXP_DIR/lr_final_full.yaml}" ;;
    validate)     pixi run python -m src.submission validate "${1:?usage: ./run.sh validate <csv>}" ;;
    submit)
        load_env
        pixi run python alfred.py submit "${1:?usage: ./run.sh submit <csv> \"<message>\"}" \
            -m "${2:?usage: ./run.sh submit <csv> \"<message>\"}"
        ;;
    submissions)
        load_env
        pixi run kaggle competitions submissions -c "$COMPETITION"
        ;;

    # --- reports -------------------------------------------------------------------
    diagnose)     pixi run python alfred.py diagnose "${1:?usage: ./run.sh diagnose <experiment name>}" ;;
    report)       pixi run report "${1:-$REPORT_CFG}" ;;

    help|-h|--help) usage ;;
    *)            echo "Unknown task: $task" >&2; echo >&2; usage >&2; exit 1 ;;
esac
