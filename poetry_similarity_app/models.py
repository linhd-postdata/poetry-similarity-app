MODELS = {
    "roberta-alberti_poetry_stanzas": {
        "model": "roberta-alberti",
        "granularity": "stanzas",
        "composition_fn": "ind_rl",
        "cofc_fn": "ICM",
        "beta_ICM": 1.2
    },
    "roberta-m_poetry_stanzas": {
        "model": "roberta-m",
        "granularity": "stanzas",
        "composition_fn": "ind_rl",
        "cofc_fn": "ICM",
        "beta_ICM": 0.911250014780804
    },
    "roberta-m_poetry_lyrics": {
        "model": "roberta-m",
        "granularity": "stanzas",
        "composition_fn": "sum",
        "cofc_fn": "cos_sim",
    },
    "roberta-alberti_poetry_lyrics": {
        "model": "roberta-alberti",
        "granularity": "stanzas",
        "composition_fn": "ind_rl",
        "cofc_fn": "ICM",
        "beta_ICM": 1.226678080167802
    },
}
