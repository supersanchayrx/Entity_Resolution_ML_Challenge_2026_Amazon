"""Launch pipeline steps as SageMaker Processing jobs.

Run from a SageMaker Studio terminal (or any machine with AWS credentials + `pip install sagemaker`):

  python sagemaker/launch.py make-sample --bucket B               # small sample -> s3://B/er2026/sample/
  python sagemaker/launch.py all --bucket B --data sample --run smoke --instance-type ml.m5.2xlarge
  python sagemaker/launch.py all --bucket B --data raw --run v1    # full run on ml.r5.8xlarge
  python sagemaker/launch.py train --bucket B --run v1b --in-runs v1 --set lgb_leaves=255

S3 layout under s3://<bucket>/<prefix>/:
  raw/{train,test}/*.tsv     challenge data (you upload this once)
  sample/{train,test}/*.tsv  output of make-sample
  work/<run>/                artifacts of a run (arrays, models, logs)
  output/<run>/              matching_results.tsv + candidate_pairs.tsv
"""
import argparse
import os
import sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ["make-sample", "all", "prepare", "block", "rerank", "features", "train", "predict"]
FIRST_STEPS = {"make-sample", "all", "prepare"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="er2026")
    ap.add_argument("--data", default="raw", help="S3 sub-prefix holding train/ and test/ (raw|sample)")
    ap.add_argument("--run", default="v1", help="name of this run (artifacts -> work/<run>/)")
    ap.add_argument("--in-runs", nargs="*", default=None,
                    help="runs whose artifacts this step reads (default: the same --run)")
    ap.add_argument("--instance-type", default="ml.r5.8xlarge")
    ap.add_argument("--volume-gb", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--role", default=None, help="SageMaker execution role ARN (auto in Studio)")
    ap.add_argument("--n-s1", type=int, default=20000, help="make-sample size")
    ap.add_argument("--set", nargs="*", default=[], help="pipeline config overrides key=value")
    ap.add_argument("--no-wait", action="store_true", help="return immediately, don't stream logs")
    args = ap.parse_args()

    # the whole code folder is uploaded; refuse to ship local data/artifacts by accident
    for d in ("work", "output", "sample", "dataset"):
        if os.path.isdir(os.path.join(CODE_ROOT, d)):
            sys.exit(f"remove or move '{d}/' out of {CODE_ROOT} first (it would be uploaded with the code)")

    import sagemaker
    from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
    from sagemaker.sklearn.estimator import SKLearn

    sess = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()
    base = f"s3://{args.bucket}/{args.prefix}"
    proc = FrameworkProcessor(
        estimator_cls=SKLearn, framework_version="1.2-1", py_version="py3", role=role,
        instance_type=args.instance_type, instance_count=1, volume_size_in_gb=args.volume_gb,
        max_runtime_in_seconds=int(args.max_hours * 3600), base_job_name=f"er2026-{args.step}",
        sagemaker_session=sess, env={"PYTHONUNBUFFERED": "1", "NUMBA_CACHE_DIR": "/tmp/numba"})

    inputs = [ProcessingInput(source=f"{base}/{args.data}/", destination="/opt/ml/processing/data",
                              input_name="data")]
    in_runs = args.in_runs if args.in_runs is not None else ([] if args.step in FIRST_STEPS else [args.run])
    work_in = []
    for i, r in enumerate(in_runs):
        dst = f"/opt/ml/processing/work_in{i}"
        inputs.append(ProcessingInput(source=f"{base}/work/{r}/", destination=dst, input_name=f"work_in{i}"))
        work_in.append(dst)

    if args.step == "make-sample":
        outputs = [ProcessingOutput(source="/opt/ml/processing/output", destination=f"{base}/sample/",
                                    output_name="sample")]
        arguments = ["make-sample", "--data", "/opt/ml/processing/data", "--out",
                     "/opt/ml/processing/output", "--work", "/tmp/work", "--n-s1", str(args.n_s1)]
    else:
        outputs = [ProcessingOutput(source="/opt/ml/processing/work", destination=f"{base}/work/{args.run}/",
                                    output_name="work"),
                   ProcessingOutput(source="/opt/ml/processing/output",
                                    destination=f"{base}/output/{args.run}/", output_name="output")]
        arguments = [args.step, "--data", "/opt/ml/processing/data", "--work", "/opt/ml/processing/work",
                     "--out", "/opt/ml/processing/output"]
        if work_in:
            arguments += ["--work-in"] + work_in
    if args.set:
        arguments += ["--set"] + args.set

    print(f"launching {args.step} on {args.instance_type}: data={base}/{args.data}/ run={args.run} "
          f"reads={in_runs} args={arguments}")
    proc.run(code="run.py", source_dir=CODE_ROOT, inputs=inputs, outputs=outputs, arguments=arguments,
             wait=not args.no_wait, logs=not args.no_wait)
    if args.no_wait:
        print("job submitted; follow it in SageMaker console > Processing jobs (logs in CloudWatch)")


if __name__ == "__main__":
    main()
