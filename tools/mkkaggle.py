"""Build notebooks/kaggle/run_kaggle.ipynb for v5.5: self-contained (the code is embedded as base64 tar.gz).

    python tools/mkkaggle.py [--out notebooks/kaggle/run_kaggle.ipynb] [--version v5.5]

Cells (v5 plan section 7): 0 version, 1 hardware, 2 inputs + scratch disk, 3 embedded code,
4 Python env, 5 profile from the hardware, 6 profiler, 7 pipeline steps, 8 report, 9 validation.
"""
import argparse
import base64
import io
import json
import os
import tarfile
import time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(CODE, "notebooks", "kaggle", "run_kaggle.ipynb"))
ap.add_argument("--version", default="v5.5")
args = ap.parse_args()

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    files = ["run.py", "requirements.txt"]
    files += [f"src/{f}" for f in sorted(os.listdir(f"{CODE}/src")) if f.endswith(".py")]
    files += [f"tools/{f}" for f in sorted(os.listdir(f"{CODE}/tools")) if f.endswith((".py", ".sh"))]
    for rel in files:
        tar.add(f"{CODE}/{rel}", arcname=rel)
B64 = base64.b64encode(buf.getvalue()).decode()
STAMP = time.strftime("%Y-%m-%d %H:%M")

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(rf"""
# ML Challenge 2026: business ER on Kaggle, **{args.version}** (built {STAMP})

The pipeline code is embedded in cell 3, so the only input to add is your **private** challenge dataset.

**When the session opens**
1. Import this notebook. This cell names the version: **{args.version}**.
2. *Settings*: **Accelerator: TPU VM** (for its CPU cores and RAM; the TPU is unused), **Internet: On**,
   *Add Input* → the private dataset, keep the notebook private.
3. Run cells 1–2 and send the hardware line.
4. Check cell 5 (`SELF_TRAIN`, `LOCO`, `EXTRA_SETS`, `FROM_STEP`), then **Run All**, or **Save & Run All
   (Commit)**. If the commit waits in a queue, run it in the open session and keep the tab open.
5. Download from the Output tab: `output/` (main file, `variants/`, `candidate_pairs.tsv`), `model/`,
   `diag/`, `profile/`, `logs/`.
6. Stop the session so it stops using TPU hours.

**v5.5**: new pair features, LightGBM re-ranker, 3 stages with fold models on the
test path, a 5% holdout scored exactly like test, per-country experts and an unconstrained model blended
for the training countries, decisions per group (countries unseen in training use the pool shaped like
them), self-training for unseen countries (variant `unseen_selftrain`). Countries are an open set:
nothing names one. With `LOCO = ['india']` the run is a leave-one-country-out measurement: **do not
upload its files**; read `loco_f05_*` in report.json and model/selftrain.json.

Checkpoints: after every step the models, reports, scores and logs are copied to `/kaggle/working`
(`STATUS.json` names the finished steps). A time guard skips extra seeds, experts, self-training and
diagnostics when the session deadline gets close, so the main file is always written first. After a
crash, set `FROM_STEP` in cell 5 to the failed step and run cells 5–9 again in the same session.
""")

code(r"""
# 1. hardware (also starts the clock of the time guard)
import os, sys, shutil, psutil, subprocess, time
os.environ['ER_T0'] = str(time.time())
ram_gb = psutil.virtual_memory().total / 2**30
cores = os.cpu_count()
print(f"HARDWARE cores={cores}  RAM={ram_gb:.0f} GB  swap={psutil.swap_memory().total / 2**30:.0f} GB")
!lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
for d in ['/kaggle/working', '/kaggle/tmp', '/tmp']:
    if os.path.isdir(d):
        print(f"{d:16} free={shutil.disk_usage(d).free / 2**30:.0f} GB")
""")

code(r"""
# 2. inputs + the biggest scratch disk (/kaggle/working holds ~20 GB: keep it for results)
import glob
DATA = '/tmp/er_data'
for split, names in [('train', ['source1', 'source2', 'source3', 'ground_truth']),
                     ('test', ['source1', 'source2', 'source3'])]:
    os.makedirs(f'{DATA}/{split}', exist_ok=True)
    for n in names:
        f = f'{split}_{n}.tsv'
        hits = sorted(glob.glob(f'/kaggle/input/**/{f}', recursive=True))
        assert hits, f'{f} not found under /kaggle/input: add your dataset as an input'
        if not os.path.exists(f'{DATA}/{split}/{f}'):
            os.symlink(hits[0], f'{DATA}/{split}/{f}')
        print(f'{f:24} <- {hits[0]}  ({os.path.getsize(hits[0]) / 2**20:.0f} MB)')
roots = [d for d in ['/kaggle/tmp', '/tmp'] if os.path.isdir(d)]
ROOT = max(roots, key=lambda d: shutil.disk_usage(d).free) + '/er'
free = shutil.disk_usage(os.path.dirname(ROOT)).free / 2**30
print('DATA =', DATA, '\nROOT =', ROOT, f'({free:.0f} GB free)')
if free < 150:
    print('WARNING: < 150 GB scratch disk; the full v5 work folder (~60 GB+) may not fit')
RES = '/kaggle/working'
""")

code(r"""
# 3. unpack the embedded pipeline code (replaces ROOT/code only: work folders of earlier runs stay)
import base64, io, tarfile
CODE_B64 = "__B64__"
shutil.rmtree(f'{ROOT}/code', ignore_errors=True); os.makedirs(ROOT, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz').extractall(f'{ROOT}/code')
print(sorted(os.listdir(f'{ROOT}/code')), sorted(os.listdir(f'{ROOT}/code/src')))
""".replace("__B64__", B64))

code(r"""
# 4. Python 3.11 env with the pinned packages (same as EC2); falls back to Kaggle's python
VENV = f'{ROOT}/venv'
rc = subprocess.run(f'pip -q install uv && uv venv -q --python 3.11 {VENV} && '
                    f'uv pip install -q --python {VENV}/bin/python -r {ROOT}/code/requirements.txt',
                    shell=True).returncode
PY = f'{VENV}/bin/python' if rc == 0 else sys.executable
if rc != 0:
    print('WARNING: pinned install failed (Internet off?) -> Kaggle python; results may differ slightly')
!{PY} -c "import numba, lightgbm, numpy, scipy, pandas; print('numba', numba.__version__, 'lgb', lightgbm.__version__, 'numpy', numpy.__version__)"
env = dict(os.environ, NUMBA_CACHE_DIR='/tmp/numba', PYTHONUNBUFFERED='1')
# parser self-test: stop here if the text rules changed behaviour
r = subprocess.run([PY, '-m', 'src.selftest'], cwd=f'{ROOT}/code', env=env, capture_output=True, text=True)
print((r.stdout + r.stderr)[-3000:])
assert r.returncode == 0, 'selftest failed (see above)'
""")

code(r"""
# 5. profile from the hardware (v5 plan section 7, v5.5 switches on top)
#   >= 150 GB and >= 64 cores: v55          (all data, a pool per unseen test country, recall push,
#                                            lr 0.04, 3 stages, 2 seeds per last-stage fold model)
#   >= 150 GB, < 64 cores:     v55_fewcores (lr 0.06, 1 seed)
#   100-150 GB:                v55_midmem   (no extra pools, no experts/unconstrained, 1.2M S1s per fit)
#   < 100 GB:                  v55_lite     (the EC2 settings, no experts/unconstrained)
import json
sys.path.insert(0, f'{ROOT}/code')
for m in [m for m in sys.modules if m == 'src' or m.startswith('src.')]:
    del sys.modules[m]                   # a rerun after a code change must not see the old module
from src.config import PROFILES, pick_profile
PROFILE = pick_profile(ram_gb, cores)
SELF_TRAIN = True        # the FAQ allows self-training on unlabeled test records (variant unseen_selftrain)
LOCO = []                # e.g. ['india']: leave-one-country-out measurement run -- do NOT upload its files
EXTRA_SETS = []          # e.g. ['country_lambda={"france":0.5}'] only with evidence for it
FROM_STEP = 'prepare'    # after a crash in this session: the failed step (earlier steps' arrays are kept)
USE_PYSPY = False        # py-spy pauses every process ~10x/s: keep off for the timed full run
n_jobs = max(2, min(cores, int(ram_gb // 4)))
DEADLINE_MIN = 500       # the time guard's deadline, minutes from cell 1 (Kaggle ends TPU sessions at ~9 h)
SETS = [f'n_jobs={n_jobs}', f'lgb_threads={n_jobs}', 'chunk_rows=500', f'deadline_min={DEADLINE_MIN}',
        f'self_train={str(SELF_TRAIN).lower()}', f'loco_countries={json.dumps(LOCO)}'] + EXTRA_SETS
if PROFILE == 'v5_fewcores':
    SETS.append('self_train=false')      # the plan's profile table: no self-training below 64 cores
print('PROFILE =', PROFILE, PROFILES[PROFILE]); print('SETS =', SETS)
""")

code(r"""
# 6. profiling helper: samples CPU / memory / IO of the whole process tree every 5 s
import threading, json, csv, collections
PROF = f'{RES}/profile'; os.makedirs(PROF, exist_ok=True)
cpu_model = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo')
                  if l.lower().startswith(('model name', 'cpu model', 'hardware')) and ':' in l), 'unknown')
PYSPY = None
if USE_PYSPY:
    subprocess.run('pip -q install py-spy', shell=True)
    PYSPY = shutil.which('py-spy')
    if PYSPY and subprocess.run([PYSPY, 'record', '-o', '/tmp/_t.txt', '--format', 'raw', '--', PY, '-c', 'pass'],
                                capture_output=True).returncode != 0:
        PYSPY = None                     # needs ptrace, which some containers forbid
print('py-spy:', 'on' if PYSPY else 'off')
HW = {'cores': cores, 'ram_gb': round(ram_gb, 1), 'cpu': cpu_model, 'profile': PROFILE, 'sets': SETS,
      'python': PY}
json.dump(HW, open(f'{PROF}/hardware.json', 'w'), indent=1); print(HW)


class Sampler(threading.Thread):
    def __init__(self, pid, path, dt=5):
        super().__init__(daemon=True)
        self.pid, self.path, self.dt = pid, path, dt
        self.stop, self.rows, self.seen = threading.Event(), [], {}

    def run(self):
        root = psutil.Process(self.pid); io0 = psutil.disk_io_counters(); t0 = time.time()
        psutil.cpu_percent(percpu=True)
        while not self.stop.wait(self.dt):
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            rss = cpu = 0.0
            for q in procs:
                try:
                    q = self.seen.setdefault(q.pid, q)
                    rss += q.memory_info().rss; cpu += q.cpu_percent()
                except psutil.Error:
                    pass
            per = psutil.cpu_percent(percpu=True); vm = psutil.virtual_memory(); io = psutil.disk_io_counters()
            r = {'t': round(time.time() - t0), 'n_procs': len(procs), 'tree_cpu_pct': round(cpu),
                 'busy_cores': sum(c > 80 for c in per), 'sys_cpu_pct': round(sum(per) / len(per)),
                 'rss_gb': round(rss / 2**30, 2), 'avail_gb': round(vm.available / 2**30, 2),
                 'swap_gb': round(psutil.swap_memory().used / 2**30, 2),
                 'disk_free_gb': round(shutil.disk_usage(ROOT).free / 2**30, 1),
                 'read_mb': round((io.read_bytes - io0.read_bytes) / 2**20),
                 'write_mb': round((io.write_bytes - io0.write_bytes) / 2**20)}
            self.rows.append(r)
            if len(self.rows) % 12 == 0:
                print(f"   [mon t={r['t']}s] busy={r['busy_cores']}/{HW['cores']} rss={r['rss_gb']}GB "
                      f"avail={r['avail_gb']}GB disk_free={r['disk_free_gb']}GB procs={r['n_procs']}", flush=True)
        with open(self.path, 'w', newline='') as f:
            if self.rows:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)

    def summary(self):
        R = self.rows or [{}]; n = max(len(self.rows), 1)
        col = lambda k: [r.get(k, 0) for r in R]
        return {'peak_rss_gb': max(col('rss_gb')), 'min_avail_gb': min(col('avail_gb')),
                'min_disk_free_gb': min(col('disk_free_gb')), 'avg_busy_cores': round(sum(col('busy_cores')) / n, 1),
                'share_time_single_core': round(sum(b <= 1 for b in col('busy_cores')) / n, 2)}
""")

code(r"""
# 7. the pipeline, step by step; each step checkpoints models/reports/scores/logs to /kaggle/working.
#    Optional work (extra seeds, self-training, diagnostics) checks the time guard itself.
STEPS = ['prepare', 'block', 'rerank', 'features', 'train', 'predict', 'variants', 'selftrain', 'diagnose']
WORK, OUT = f'{ROOT}/work55', f'{RES}/output'
timings = {}
for step in STEPS[STEPS.index(FROM_STEP):]:
    t0 = time.time(); print(f'== {step}: running ({time.strftime("%H:%M:%S")})', flush=True)
    cmd = [PY, 'run.py', step, '--data', DATA, '--work', WORK, '--out', OUT, '--checkpoint', RES,
           '--profile', PROFILE, '--set', *SETS]
    if PYSPY:
        cmd = [PYSPY, 'record', '--rate', '10', '--subprocesses', '--native', '--format', 'raw',
               '-o', f'{PROF}/pyspy_{step}.txt', '--'] + cmd
    p = subprocess.Popen(cmd, cwd=f'{ROOT}/code', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    mon = Sampler(p.pid, f'{PROF}/{step}_resources.csv'); mon.start()
    with open(f'{PROF}/{step}_stdout.log', 'w') as flog:
        for line in p.stdout:
            print(line, end=''); flog.write(line); flog.flush()
    rc = p.wait(); mon.stop.set(); mon.join()
    timings[step] = {'minutes': round((time.time() - t0) / 60, 1), 'rc': rc, **mon.summary()}
    json.dump(timings, open(f'{PROF}/summary.json', 'w'), indent=1)
    print(f'== {step}: {json.dumps(timings[step])}', flush=True)
    if rc != 0 and step in ('selftrain', 'diagnose'):
        print(f'WARNING: optional step {step} failed; the main file and variants are already written')
        continue
    assert rc == 0, f'{step} failed (see log above)'
""")

code(r"""
# 8. profile report
print(f"{'step':9} {'min':>6} {'busy/cores':>11} {'1-core%':>8} {'peakRSS':>8} {'minFreeDisk':>11}")
for k, v in timings.items():
    print(f"{k:9} {v['minutes']:6.1f} {v['avg_busy_cores']:5}/{HW['cores']:<5} "
          f"{100 * v['share_time_single_core']:7.0f}% {v['peak_rss_gb']:7.1f}G {v['min_disk_free_gb']:10.0f}G")
for step in timings:
    f = f'{PROF}/pyspy_{step}.txt'
    if not os.path.exists(f):
        continue
    self_t, tot = collections.Counter(), 0
    for line in open(f):
        stack, _, n = line.rstrip().rpartition(' ')
        if n.isdigit():
            n = int(n); tot += n; self_t[stack.split(';')[-1][:110]] += n
    print(f'\n--- {step}: top self-time frames ({tot} samples) ---')
    for fr, n in self_t.most_common(12):
        print(f'{100 * n / max(tot, 1):5.1f}%  {fr}')
""")

code(r"""
# 9. validation + summary
!{PY} {ROOT}/code/tools/validate_submission.py -m {OUT}/matching_results.tsv -c {OUT}/candidate_pairs.tsv -t {DATA}/test
!cat {OUT}/variants/summary.tsv
!cat {RES}/model/report.json
!cat {RES}/model/selftrain.json 2>/dev/null
!cat {RES}/STATUS.json
!ls {OUT}/variants/*/DRIFT 2>/dev/null && echo 'a variant is flagged DRIFT: do not upload it'
rep = json.load(open(f'{RES}/model/report.json'))
for k in sorted(rep):
    if k.startswith(('oof_f05', 'holdout_f05', 'loco_f05')):
        print(f'{k:40} {rep[k]:.5f}' if isinstance(rep[k], float) else f'{k:40} {rep[k]}')
# per-country health without labels (expected F0.5 from the model's own probabilities)
r = subprocess.run([PY, 'tools/country_health.py', '--work', WORK, '--profile', PROFILE, '--set', *SETS],
                   cwd=f'{ROOT}/code', env=env, capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
if os.path.exists(f'{WORK}/diag/country_health.tsv'):
    os.makedirs(f'{RES}/diag', exist_ok=True); shutil.copy(f'{WORK}/diag/country_health.tsv', f'{RES}/diag/')
if LOCO:
    print('LOCO run: a measurement, not a submission. Do not upload these files.')
!du -sh {RES}/* {WORK}
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", args.out, f"({len(B64) / 1024:.0f} KB embedded code)")
