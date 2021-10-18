MODELS = {
    "roberta-alberti_poetry_stanzas": {
        "model": "alberti",
        "granularity": "stanzas",
        "similarity_fn": "ind_rl",
        "composition_fn": "ICM",
        "beta_ICM": 1.2
    },
    "roberta-m_poetry_stanzas": {
        "model": "roberta-m",
        "granularity": "stanzas",
        "similarity_fn": "ind_rl",
        "composition_fn": "ICM",
        "beta_ICM": 0.9
    },
    "roberta-m__poetry_stanzas": {
        "model": "roberta-m",
        "granularity": "stanzas",
        "similarity_fn": "sum",
        "composition_fn": "cos_sim",
    },
    "roberta-alberti_poetry_lyrics": {
        "model": "roberta-m",
        "granularity": "stanzas",
        "similarity_fn": "ind_rl",
        "composition_fn": "ICM",
        "beta_ICM": 1.226678080167802
    },
}
