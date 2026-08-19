# Environment Locks

Store realized environment snapshots here after setup succeeds. These files are
records of a working machine, not universal cross-platform locks. Labels are
write-once, so use a new label for each distinct environment record.

Example:

```bash
python scripts/record_environment.py pennylane-env
```
