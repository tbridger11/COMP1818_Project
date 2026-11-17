## Work of Toby Michael Bridger
### email: tb9509i@gre.ac.uk
### Coursework for COMP1818, Term 1, Submission 18th Nov 2026.

# COMP1818 MiniGrid Generalisation Project

This project compares PPO models trained on fixed and mixed grid sizes, including
a simplified PLR-inspired size sampler, using the existing evaluation and
plotting pipeline.

## Setup

Use Python 3.13 and run commands from the repository root.

~~~powershell
git clone https://github.com/tbridger11/COMP1818_Project.git
cd COMP1818_Project
python -m venv venv
python -m pip install -r requirements.txt
~~~

The repository already contains all trained models and evaluation results. Training experiments do not need to be rerun to reproduce the reported figures and tables.

TensorBoard logs and generated plots/tables are local outputs and are excluded
from Git.

# Running Scripts
All scripts should be run from the repository root using Python modules:

~~~powershell
python -m src.training.ppo_baseline
python -m src.training.ppo_mixed
python -m src.training.ppo_mixed_optuna
python -m src.training.ppo_plr

python -m src.evaluation.evaluate_ppo
python -m src.evaluation.evaluate_random
python -m src.evaluation.evaluate_random_nav_only

python -m src.plotting.plot_results
~~~

Scripts are designed to be executed as modules rather than by running individual .py files directly. This ensures package imports and relative paths resolve correctly across different IDEs and environments.

## Project layout

- `src\environments`: custom MiniGrid environments.
- `src\training`: PPO training and Optuna tuning.
- `src\evaluation`: trained-model and random-baseline evaluation.
- `src\plotting`: plots and result tables.
- `src\tools`: interactive viewers and environment checks.

## Train models

Run the training scripts you need:

~~~powershell
python -m src.training.ppo_baseline
python -m src.training.ppo_15x15_optuna
python -m src.training.ppo_mixed
python -m src.training.ppo_mixed_optuna
python -m src.training.ppo_plr
~~~

All PPO training scripts are configured for the final 250k runs with seeds
42, 123, and 999:

- `ppo_baseline`: fixed-size baselines at 5x5, 10x10, 15x15, and 20x20.
- `ppo_15x15_optuna`: Optuna-tuned fixed 15x15 PPO.
- `ppo_mixed`: mixed-size PPO, 250k timesteps.
- `ppo_mixed_optuna`: Optuna-tuned mixed-size PPO.
- `ppo_plr`: PLR-inspired PPO.

Each script uses `TIMESTEPS = 250_000` and training seeds 42, 123, and 999.
Model names derive their timestep suffix from `TIMESTEPS`.
Commented 500k options are kept for the saved 500k comparison groups. In
`ppo_baseline.py`, switch `TRAIN_SIZES` to `[15]` as well to reproduce the
500k fixed 15x15 models. `ppo_plr.py` also keeps the saved 50k
validation configuration commented out (seeds 42 and 123). Model names update
their timestep suffix automatically.

These settings and commented options cover the saved PPO model groups, except
the two test models under `saved models\test`. The 250k 15x15 Optuna run is also
configured for the final comparison.

Run the scripts you want from the repository root. They save models under
`saved models` and metadata under `results\training`.

The 15x15 Optuna model uses the same tuned PPO parameters as the mixed Optuna
trainer. If tuning is rerun, update those values in both trainers and the PLR
trainer.

To retune mixed PPO, run:

~~~powershell
python -m src.training.tune_mixed_ppo
~~~

## Evaluate models

`src\evaluation\evaluate_ppo.py` evaluates one model group per run. In that file, enable the
configuration matching the trained model and disable the active configuration.
The active preset evaluates the 250k fixed-size Optuna model. Other 250k
presets, including PLR, are available in the file.

Evaluation uses training seeds 42, 123, and 999; test sizes 5, 10, 15, 20, 30,
and 40; and 250 fixed evaluation seeds (1000–1249) per size.

~~~powershell
python -m src.evaluation.evaluate_ppo
~~~

Results append to `results\evaluations\evaluation_results.csv`. Move or remove
that file before a clean rerun to avoid duplicate rows.

## Random baselines and plots

Run the random evaluations if needed:

~~~powershell
python -m src.evaluation.evaluate_random
python -m src.evaluation.evaluate_random_nav_only
~~~

Generate plots and tables from the evaluation CSV:

~~~powershell
python -m src.plotting.plot_results
~~~

Outputs are saved under `plots_all_models` and `tables_all_models`. The 250k
plots include PLR and the 15x15 Optuna model. The 500k comparison chart is for
previous 500k runs and includes the fixed-size PPO, Optuna-tuned fixed 15x15
PPO, mixed Optuna PPO, and PLR-inspired PPO. Each model needs evaluation
results for all six test sizes.

## TensorBoard

~~~powershell
tensorboard --logdir logs\tensorboard
~~~

Open http://localhost:6006 to view training logs.

## Acknowledgements
This project builds upon several open-source reinforcement learning projects:

MiniGrid
Gymnasium
Stable-Baselines3
Optuna
The project uses these frameworks under their respective licences. Custom environments, training pipelines, evaluation procedures and experiments were developed as part of COMP1818 coursework.

## AI assistance

AI tools assisted with drafting and editing this README to ensure there would be no isssues with running this locally. 
