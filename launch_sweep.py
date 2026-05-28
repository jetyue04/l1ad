"""Generate and optionally apply one NRP job per (x_step, x_step_size) combination.

Usage:
    python launch_sweep.py                  # print kubectl commands only
    python launch_sweep.py --apply          # also run kubectl apply for each job
    python launch_sweep.py --dry-run        # just print the first job YAML
"""

import argparse
import itertools
import subprocess
import tempfile
from pathlib import Path

# ── Sweep grid ────────────────────────────────────────────────────────────────
X_STEPS      = [5, 10, 20, 50]
X_STEP_SIZES = [0.01, 0.05, 0.1, 0.2]

# ── NRP settings (edit to match your cluster) ─────────────────────────────────
NAMESPACE       = "axol1tl"
IMAGE           = "gitlab-registry.nrp-nautilus.io/mquinnan/axol1tl-hub:axol1tl-container"
PVC_NAME        = "axovol"
PVC_MOUNT       = "/axovol"
WORKDIR         = "/axovol/l1ad"
VICREG_CKPT     = "/axovol/l1ad/checkpoints/vicreg_fixed/checkpoint_epoch1000.pt"
SWEEP_BASE_DIR  = "/axovol/l1ad/checkpoints/sweep"
CPU             = "4"
MEMORY          = "16Gi"
GPUS            = 1


def job_name(x_step, x_step_size):
    size_str = f"{x_step_size:.4f}".replace(".", "p").rstrip("0")
    return f"wnae-sweep-s{x_step}-ss{size_str}"


def job_yaml(x_step, x_step_size):
    name    = job_name(x_step, x_step_size)
    outdir  = f"{SWEEP_BASE_DIR}/step{x_step}_size{x_step_size}"
    log_file = f"{outdir}/training.log"
    cmd = (
        f"set -e && "
        f"cd {WORKDIR} && "
        f"git pull && "
        f"pip install pot reportlab --quiet && "
        f"mkdir -p {outdir} && "
        f"python -u sweep_train_wnae.py "
        f"--vicreg-checkpoint {VICREG_CKPT} "
        f"--wnae-config config/vicreg_wnae_config.yaml "
        f"--x-step {x_step} "
        f"--x-step-size {x_step_size} "
        f"--outdir {outdir} "
        f"| tee {log_file}"
    )

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NAMESPACE}
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: {IMAGE}
          imagePullPolicy: Always
          volumeMounts:
            - name: axovol
              mountPath: {PVC_MOUNT}
          workingDir: {WORKDIR}
          command:
            - "/bin/bash"
            - "-c"
            - |
              {cmd}
          resources:
            requests:
              cpu: "{CPU}"
              memory: "{MEMORY}"
              nvidia.com/gpu: {GPUS}
            limits:
              cpu: "{CPU}"
              memory: "{MEMORY}"
              nvidia.com/gpu: {GPUS}
      volumes:
        - name: axovol
          persistentVolumeClaim:
            claimName: {PVC_NAME}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply",   action="store_true", help="Run kubectl apply for each job")
    parser.add_argument("--dry-run", action="store_true", help="Print first job YAML and exit")
    args = parser.parse_args()

    combos = list(itertools.product(X_STEPS, X_STEP_SIZES))
    print(f"Sweep: {len(X_STEPS)} x_steps × {len(X_STEP_SIZES)} x_step_sizes = {len(combos)} jobs\n")

    for x_step, x_step_size in combos:
        name = job_name(x_step, x_step_size)
        yaml = job_yaml(x_step, x_step_size)

        if args.dry_run:
            print(yaml)
            return

        print(f"Job: {name}")

        if args.apply:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                f.write(yaml)
                tmp = f.name
            result = subprocess.run(["kubectl", "apply", "-f", tmp], capture_output=True, text=True)
            print(f"  {result.stdout.strip() or result.stderr.strip()}")
            Path(tmp).unlink()
        else:
            outdir = f"{SWEEP_BASE_DIR}/step{x_step}_size{x_step_size}"
            print(f"  x_step={x_step}  x_step_size={x_step_size}  -> {outdir}")

    if not args.apply:
        print("\nRun with --apply to submit all jobs, or --dry-run to inspect a single YAML.")


if __name__ == "__main__":
    main()
