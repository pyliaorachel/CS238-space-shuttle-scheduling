# CS238 Space SHuttle Scheduling

## Usage

```bash
# Generate batch experiences
$ python -m src.batch_experiences <csv-output-file-path>

# Train & evaluate
$ python -m src.run --algo <algorithm> --n-episodes <num-of-episodes>

# Human evaluation
$ python -m src.human_eval
```

## Algorithms

- `random`
- `q-learning`
- `mlrl`
- `pomdp`
