# CS238 Space SHuttle Scheduling

## Usage

```bash
# Generate batch experiences
$ python -m src.batch_experiences <csv-output-file-path>

# Train
$ python -m src.train --algo <algorithm> --n-episodes <num-of-episodes>
```

## Algorithms

- `random`
- `q-learning`
- `mle-vi`
- `pomdp`
