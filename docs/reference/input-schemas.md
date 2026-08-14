# Input schemas

Public validators return canonical copies and reject ambiguous schemas. Run
them before an expensive inference or scan.

## Donor-ignorant event counts

```csv
cell_id,count
cell_001,0
cell_002,3
cell_003,1
```

- columns must be exactly `cell_id,count` for `validate_count_frame`;
- cell identifiers must be non-empty and unique;
- counts must be finite, non-negative integers;
- the shared observation time is supplied separately and must be positive.

## Donor-aware counts

```csv
cell_id,donor_id,count
cell_001,donor_A,0
cell_002,donor_A,3
cell_003,donor_B,1
```

Donor identifiers must be non-empty. The validator preserves donor ordering so
posterior donor coordinates can be mapped back to labels.

## Multiple conditions

```csv
cell_id,condition,count
cell_001,Control,0
cell_002,Control,2
cell_003,Treatment,4
```

For donor-aware condition fits, use:

```csv
cell_id,donor_id,condition,count
cell_001,donor_A,Control,0
cell_002,donor_A,Control,2
cell_003,donor_B,Treatment,4
```

`normalize_condition_frame` can canonicalize accepted aliases;
`validate_condition_frame` applies strict scientific constraints; and
`split_condition_frame` returns per-condition count/donor frames. Conditions
are fitted independently.

## Compact trajectory form

```csv
cell_id,condition,history
cell_001,Control,"0,0,1,0"
cell_002,Control,"1,1"
cell_003,Control,""
```

`0` is non-lethal and `1` is lethal. A blank history retains a zero-contact
cell. Do not substitute a missing identifier or condition with a blank value.

## Wide and long trajectory forms

The normalizer also accepts wide event columns or a long representation with
one row per contact. A long table identifies the cell, condition, contact order,
and binary outcome. `normalize_trajectory_frame` converts supported forms to
compact canonical histories; `expanded_trajectory_frame` converts canonical
histories back to one row per observed contact.

Use `read_trajectory_csv` for bytes or text. It rejects malformed CSV and then
applies the same normalization rules.

## Package-scale safety ceilings

The standalone library uses broad guards against accidental pathological
allocations—not recommended workload sizes. Current package ceilings allow up
to approximately one million cells, one million trajectory SMC particles, 128
chains, and 1,000 conditions where applicable. A request can pass these guards
and still be computationally impractical by orders of magnitude.

Frontends, hosted services, notebooks, and institutional deployments should
impose much tighter limits based on available memory, runtime, queue capacity,
security review, and intended use. Always estimate one small fit before a grid.

## Privacy

Use anonymous study identifiers. Do not put names, clinical identifiers, dates,
raw microscopy, unapproved donor metadata, or private file paths into public
examples, Pages builds, CI logs, archives, or issue reports.
