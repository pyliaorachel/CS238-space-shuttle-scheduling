# CS238 Space Shuttle Scheduling

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
- `mlrl-ru`
- `mlrl-pu`
- `q-learning`
- `sarsa`
