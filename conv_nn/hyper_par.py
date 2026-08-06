import subprocess
import shutil
import optuna
import sys
import os
import numpy as np
sys.path.append("../data")
from data_preprocess import simulation_data


# ==========================================================
# Train CNN
# ==========================================================
def training(trial):

    print("=" * 100)
    print(f"Trial {trial.number}: Training CNN with suggested hyperparameters...")
    print("=" * 100)

    alpha_emiss = trial.suggest_float(
        "alpha_emiss", 1, 1000, log=True
    )

    alpha_profile = trial.suggest_float(
        "alpha_profile", 1, 1000, log=True
    )

    alpha_gate = trial.suggest_float(
        "alpha_gate", 0.1, 20, log=True
    )

    alpha_leak = trial.suggest_float(
        "alpha_leak", 0.1, 50, log=True
    )

    alpha_active_pdf = trial.suggest_float(
        "alpha_active_pdf", 0.1, 50, log=True
    )

    print(f"Trial {trial.number}: Hyperparameters used:")
    print(f"alpha_emiss: {alpha_emiss}")
    print(f"alpha_profile: {alpha_profile}")
    print(f"alpha_gate: {alpha_gate}")
    print(f"alpha_leak: {alpha_leak}")
    print(f"alpha_active_pdf: {alpha_active_pdf}")

    subprocess.run(
        [
            sys.executable,
            "pdf_cnn.py",
            "--alpha_emiss", str(alpha_emiss),
            "--alpha_profile", str(alpha_profile),
            "--alpha_gate", str(alpha_gate),
            "--alpha_leak", str(alpha_leak),
            "--alpha_active_pdf", str(alpha_active_pdf),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("=" * 100)
    print(f"Trial {trial.number}: Training completed. Running Athena-K simulation...")
    print("=" * 100)

    return {
        "alpha_emiss": alpha_emiss,
        "alpha_profile": alpha_profile,
        "alpha_gate": alpha_gate,
        "alpha_leak": alpha_leak,
        "alpha_active_pdf": alpha_active_pdf,
    }


# ==========================================================
# Run Athena-K
# ==========================================================
def run_athena(trial):

    print("=" * 100)
    print(f"Trial {trial.number}: Running Athena-K simulation...")
    print("=" * 100)

    athena_dir = "../AthenaK_legacy/sg_build/src"

    folder_name = f"pdf_trial_{trial.number}"

    cmd = [
        "./athena",
        "-i", "sg.athinput",
        "-d", folder_name,
        "-r", "lrct16_8/rst/KH.00005.rst",
    ]

    result = subprocess.run(
        cmd,
        cwd=athena_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print(f"Athena exited with code {result.returncode}")

    folder = os.path.join(athena_dir, folder_name)

    if not os.path.exists(os.path.join(folder, "cool_rate.bin")):
        raise RuntimeError("Athena failed before producing output.")

    print("=" * 100)
    print(f"Trial {trial.number}: Athena-K simulation completed. Results saved in folder: {folder_name}")
    print("=" * 100)

    return folder_name


# =========================================================
# Compare simulation results with high-resolution reference
# =========================================================
def compare_simulation(trial, delete_folder=True):

    print("=" * 100)
    print(f"Trial {trial.number}: Comparing simulation results with high-resolution reference...")
    print("=" * 100)

    HR_EMISSIVITY = 1.47
    HR_MASS_FLUX = -0.451

    gamma = 1.6667
    rho0 = 1e-3
    p0 = 8.63359
    du = 31.0918

    sim_data = simulation_data()
    sim_data.resolution = (16, 8)


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"pdf_trial_{trial.number}"
    file_path = os.path.abspath(
        os.path.join(
            BASE_DIR,
            "../AthenaK_legacy/sg_build/src",
            folder_name,
            "bin",
        )
    )   

    sim_data.input_data(file_path, start=501)

    rho = sim_data.rho
    temp = sim_data.temp
    uy = sim_data.uy

    cool_filename = os.path.join(
        os.path.dirname(file_path),
        "cool_rate.bin"
    )

    nx = rho.shape[2]
    ny = rho.shape[1]

    cool = np.fromfile(
        cool_filename,
        dtype=np.float64
    ).reshape(-1, ny, nx)

    emiss = cool / (p0 * du)

    emiss_xavg = emiss.mean(axis=2)

    emiss_mean = emiss_xavg.mean(axis=0)

    y = np.linspace(
        0,
        sim_data.total_length,
        ny
    )

    int_emiss = np.trapz(emiss_mean, y)

    mass_flux = np.mean(
        (rho * uy)[:, -1, :]
    ) / (rho0 * du)

    emiss_err = abs(int_emiss - HR_EMISSIVITY) / abs(HR_EMISSIVITY)

    mass_err = abs(mass_flux - HR_MASS_FLUX) / abs(HR_MASS_FLUX)

    score = emiss_err + mass_err

    print("=" * 100)
    print(f"Trial {trial.number}: Integrated emissivity : {int_emiss:.5f}")
    print(f"Trial {trial.number}: Mass flux             : {mass_flux:.5f}")
    print(f"Trial {trial.number}: Objective             : {score:.6f}")
    print("=" * 100)

    folder_path = os.path.join(
        BASE_DIR,
        "../AthenaK_legacy/sg_build/src",
        folder_name,
    )   

    if delete_folder:
        shutil.rmtree(folder_path)

    return score

class DummyTrial:
    def __init__(self, params, number=0):
        self._trial = optuna.trial.FixedTrial(params)
        self.number = number

    def suggest_float(self, *args, **kwargs):
        return self._trial.suggest_float(*args, **kwargs)

    def suggest_int(self, *args, **kwargs):
        return self._trial.suggest_int(*args, **kwargs)

    def suggest_categorical(self, *args, **kwargs):
        return self._trial.suggest_categorical(*args, **kwargs)

def objective(trial):

    training(trial)

    run_athena(trial)

    score = compare_simulation(trial)

    return score

# Dummy trial run for testing purposes
# if __name__ == "__main__":

#     trial = DummyTrial(
#         {
#             "alpha_emiss": 10.0,
#             "alpha_profile": 10.0,
#             "alpha_gate": 1.0,
#             "alpha_leak": 10.0,
#             "alpha_active_pdf": 20.0,
#         },
#         number=0,
#     )

#     try:
#         training(trial)
#         run_athena(trial)
#         score = compare_simulation(trial)
#         print(f"\nFinal score = {score:.6f}")

#     except Exception as e:
#         print(f"\nPipeline failed:\n{e}")
#         raise

# Run over 100 iterations of hyperparameter optimization using Optuna
if __name__ == "__main__":

    study = optuna.create_study(
        study_name="cnn_hyperparams",
        storage="sqlite:///cnn_hyperparams.db",
        load_if_exists=True,
        direction="minimize",
    )

    print(f"Completed trials before optimization: {len(study.trials)}")

    study.optimize(
        objective,
        n_trials=100,
    )

    print("=" * 100)
    print("Optimization complete")
    print("=" * 100)

    print(f"Best score: {study.best_value:.6f}")

    print("\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print("\nBest trial:", study.best_trial.number)

    print("=" * 100)
    print("Re-running the best trial...")
    print("=" * 100)

    best_trial = DummyTrial(
        study.best_params,
        number=study.best_trial.number,
    )

    training(best_trial)

    run_athena(best_trial)

    final_score = compare_simulation(best_trial)

    print("=" * 100)
    print("Best trial re-run completed.")
    print(f"Final score: {final_score:.6f}")
    print("=" * 100)


# Re-run the N-th best trial from the saved Optuna study
# if __name__ == "__main__":

#     N = 3      

#     study = optuna.load_study(
#         study_name="cnn_hyperparams",
#         storage="sqlite:///cnn_hyperparams.db",
#     )

#     # Keep only completed trials
#     completed_trials = [
#         t for t in study.trials
#         if t.state == optuna.trial.TrialState.COMPLETE
#     ]

#     # Sort by objective (lowest is best)
#     completed_trials.sort(key=lambda t: t.value)

#     if N < 1 or N > len(completed_trials):
#         raise ValueError(
#             f"Only {len(completed_trials)} completed trials are available."
#         )

#     selected = completed_trials[N - 1]

#     print("=" * 100)
#     print(f"Re-running the #{N} best trial")
#     print(f"Original trial number : {selected.number}")
#     print(f"Objective value       : {selected.value:.6f}")
#     print("=" * 100)

#     print("\nHyperparameters:")
#     for k, v in selected.params.items():
#         print(f"{k}: {v}")

#     trial = DummyTrial(
#         selected.params,
#         number=selected.number,
#     )

#     training(trial)

#     run_athena(trial)

#     score = compare_simulation(trial, delete_folder=False)

#     print("=" * 100)
#     print("Finished re-running selected trial")
#     print(f"Objective score: {score:.6f}")
#     print("=" * 100)