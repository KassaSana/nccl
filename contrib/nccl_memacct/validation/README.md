# GPU validation run

`kaggle_validate.sh` does the following:
- builds NCCL (`v2.32.3-1` by default, the version used in the #2527 survey) for the GPUs present;
- builds `../harness.cc`;
- runs it with 1 GPU, and with 2 if available, each with `NCCL_RUNTIME_CONNECT=1` and `0`;
- runs `../nccl_memacct.py` on each log.

It records the GPU model, compute capability, driver, CUDA toolkit, compiler, OS, NCCL commit and the NCCL
version line from the log in `environment.txt` and `summary.md`. Report results together with that
environment. Results from another NCCL version or GPU aren't directly comparable with the survey's numbers.

## On Kaggle

1. Zip the whole `contrib/nccl_memacct/` folder, which includes `validation/`, and upload the zip as a
   private Kaggle dataset.
2. Create a notebook, attach the dataset, and set **Accelerator → GPU T4 x2**. GPU quota, driver and CUDA
   version vary, and the script records whatever it finds.
3. Kaggle may add an extra folder level when it unpacks the zip, so locate the script instead of guessing
   its path, then run it:
   ```
   %%bash
   SCRIPT="$(find /kaggle/input -name kaggle_validate.sh | head -1)"
   echo "Using $SCRIPT"
   bash "$SCRIPT"
   ```
   The script finds `harness.cc` and `nccl_memacct.py` relative to its own location. The dataset is mounted
   read-only, which is fine, because all output goes to `/kaggle/working`.
4. Download `/kaggle/working/memacct_results.tar.gz`.

The NCCL build takes about 10 minutes or more. The script reuses an existing build when you run it again in
the same session. If the build fails, it stops and the log in `memacct_results/` says why. Set
`NCCL_REF=<tag>` to try another version.

## Limits on this hardware

T4 VMs have no NVLink, so this run doesn't cover the NVLS or MNNVL paths. It also only uses one process.
Whether P2P or SHM gets used depends on the VM; the log and `nvidia-smi topo -m` in `environment.txt` show
which.
