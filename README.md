ACTA/
│
├── main.py
├── trainers.py
│
├── algorithms/
│   ├── __init__.py
│   ├── algorithms_base.py
│   ├── ACON.py
│   ├── ACTA.py
│   └── components/
│       ├── __init__.py
│       ├── acta_aligner.py
│       └── semantic_teacher.py
│
├── dataloader/
│   ├── __init__.py
│   └── dataloader.py
│
├── configs/
│   └── data_model_configs.py
│
├── utils/
│   ├── module.py
│   ├── loss.py
│   └── plot.py
│
├── semantic_preparation/
│   ├── __init__.py
│   ├── build_semantic_package.py
│   ├── dtw_utils.py
│   └── local_admissibility.py
│
├── semantic_packages/
│   └── ...
│
├── source_models/
│   └── ...
│
├── scripts/
│   ├── prepare_source.py
│   ├── prepare_semantics.py
│   ├── smoke_test.py
│   └── run_pilot.py
│
└── README.md