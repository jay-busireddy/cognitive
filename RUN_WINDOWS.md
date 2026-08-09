# Windows laptop run instructions

Open PowerShell in the extracted package directory.

## 1. Create a clean environment

```powershell
py -m venv .venv_cogval
.\.venv_cogval\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 2. Syntax check

```powershell
python -m py_compile cognitive_validation.py
python -m py_compile resource_benchmark.py
```

## 3. Smoke test - software verification only

```powershell
python cognitive_validation.py --preset smoke --mode all --output cognitive_smoke
```

Do not use smoke p-values in a paper.

## 4. Optional engineering pilot

```powershell
python cognitive_validation.py --preset pilot --mode all --output cognitive_pilot
```

Use the pilot only to catch implementation/runtime issues. If you change parameters after looking at the pilot, the pilot is exploratory and must not be merged with the final confirmatory p-values.

## 5. Final confirmatory run

Run the 18 core hypotheses:

```powershell
python cognitive_validation.py --preset confirmatory --mode core --output cognitive_confirmatory_core
```

Then, if desired, run the four proxy constructs separately:

```powershell
python cognitive_validation.py --preset confirmatory --mode all --output cognitive_confirmatory_all
```

For the cleanest paper, I recommend using `cognitive_confirmatory_core` as the confirmatory family and using the proxy results only as secondary/exploratory evidence. If you already ran `--mode all` first, the code still reports Holm correction for the core family separately.

## 6. Local resource-scaling benchmark

```powershell
python resource_benchmark.py --output cognitive_resource_benchmark
```

This reports machine-specific representation memory and query latency. Do not compare its numbers directly to earlier paper figures unless the underlying data structures/workload are equivalent.

## 7. Zip the final outputs

```powershell
Compress-Archive -Path .\cognitive_confirmatory_core\* -DestinationPath .\cognitive_confirmatory_core.zip
Compress-Archive -Path .\cognitive_resource_benchmark\* -DestinationPath .\cognitive_resource_benchmark.zip
```

Upload both ZIP files for an independent statistical audit and empirical-paper drafting.

## Do not do this

- Do not repeatedly rerun with new seeds until p < .05.
- Do not remove a failed hypothesis after seeing the result.
- Do not tune weights on the confirmatory seeds.
- Do not call p >= .05 proof of the null.
- Do not call a proxy test evidence of consciousness, genuine empathy, or moral correctness.

