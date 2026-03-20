from dataclasses import dataclass


@dataclass
class ToyGenerator:
    scale: float = 1.0
    bias: float = 0.0

    def forward(self, x):
        return [max(0.0, min(1.0, self.scale * v + self.bias)) for v in x]

    def n_params(self):
        return 2


@dataclass
class ToyDiscriminator:
    weight: float = 1.0
    bias: float = 0.0

    def score(self, x):
        scores = []
        for v in x:
            s = self.weight * v + self.bias
            s = max(0.0, min(1.0, s))
            scores.append(s)
        return scores

    def n_params(self):
        return 2


class ToyCycleGAN:
    def __init__(
        self,
        name="toy_resnet",
        hidden_size=64,
        use_reflection_pad=False,
        reflection_bonus=0.02,
        init_g_ab=(0.9, 0.05),
        init_g_ba=(1.05, -0.02),
        structural_multiplier=32,
        optimization_profile=None,
        model_score_bonus=0.0,
    ):
        bonus = float(reflection_bonus) if use_reflection_pad else 0.0
        self.name = name
        self.g_ab = ToyGenerator(scale=float(init_g_ab[0]) + bonus, bias=float(init_g_ab[1]))
        self.g_ba = ToyGenerator(scale=float(init_g_ba[0]) + bonus, bias=float(init_g_ba[1]))
        self.d_a = ToyDiscriminator(weight=1.0, bias=0.0)
        self.d_b = ToyDiscriminator(weight=1.0, bias=0.0)
        self.hidden_size = hidden_size
        self.use_reflection_pad = use_reflection_pad
        self.reflection_bonus = float(reflection_bonus)
        self.structural_multiplier = int(structural_multiplier)
        self.optimization_profile = optimization_profile or {
            "gen_scale_gain": 8.0,
            "gen_bias_gain": 2.0,
            "disc_gain": 4.0,
        }
        self.model_score_bonus = float(model_score_bonus)

    def parameter_count(self):
        structural = int(self.hidden_size) * int(self.structural_multiplier)
        return structural + self.g_ab.n_params() + self.g_ba.n_params() + self.d_a.n_params() + self.d_b.n_params()

    def to_state_dict(self):
        return {
            "name": self.name,
            "g_ab": vars(self.g_ab),
            "g_ba": vars(self.g_ba),
            "d_a": vars(self.d_a),
            "d_b": vars(self.d_b),
            "hidden_size": self.hidden_size,
            "use_reflection_pad": self.use_reflection_pad,
            "reflection_bonus": self.reflection_bonus,
            "structural_multiplier": self.structural_multiplier,
            "optimization_profile": self.optimization_profile,
            "model_score_bonus": self.model_score_bonus,
        }


MODEL_SPECS = {
    "toy_resnet": {
        "init_g_ab": (0.9, 0.05),
        "init_g_ba": (1.05, -0.02),
        "structural_multiplier": 32,
        "optimization_profile": {"gen_scale_gain": 8.0, "gen_bias_gain": 2.0, "disc_gain": 4.0},
        "model_score_bonus": 0.0,
    },
    "toy_unet": {
        "init_g_ab": (0.82, 0.07),
        "init_g_ba": (1.12, -0.03),
        "structural_multiplier": 40,
        "optimization_profile": {"gen_scale_gain": 10.0, "gen_bias_gain": 3.0, "disc_gain": 3.0},
        "model_score_bonus": 0.01,
    },
    "toy_mobilenet": {
        "init_g_ab": (0.95, 0.03),
        "init_g_ba": (1.02, -0.01),
        "structural_multiplier": 24,
        "optimization_profile": {"gen_scale_gain": 7.0, "gen_bias_gain": 1.8, "disc_gain": 4.5},
        "model_score_bonus": 0.005,
    },
    "toy_dense": {
        "init_g_ab": (0.86, 0.06),
        "init_g_ba": (1.10, -0.025),
        "structural_multiplier": 48,
        "optimization_profile": {"gen_scale_gain": 9.5, "gen_bias_gain": 2.8, "disc_gain": 3.2},
        "model_score_bonus": 0.012,
    },
    "toy_transformer": {
        "init_g_ab": (0.88, 0.045),
        "init_g_ba": (1.08, -0.018),
        "structural_multiplier": 56,
        "optimization_profile": {"gen_scale_gain": 11.0, "gen_bias_gain": 3.4, "disc_gain": 2.8},
        "model_score_bonus": 0.015,
    },
    "toy_shallow": {
        "init_g_ab": (0.98, 0.02),
        "init_g_ba": (1.01, -0.008),
        "structural_multiplier": 18,
        "optimization_profile": {"gen_scale_gain": 6.2, "gen_bias_gain": 1.4, "disc_gain": 5.0},
        "model_score_bonus": -0.004,
    },
}


def supported_models():
    return sorted(MODEL_SPECS.keys())


def build_model(model_cfg):
    model_cfg = model_cfg or {}
    name = str(model_cfg.get("name", "toy_resnet")).strip().lower()
    hidden_size = int(model_cfg.get("hidden_size", 64))
    use_reflection_pad = bool(model_cfg.get("use_reflection_pad", False))
    reflection_bonus = float(model_cfg.get("reflection_bonus", 0.02))

    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"Unsupported model.name '{name}'. Supported models: {', '.join(supported_models())}"
        )

    return ToyCycleGAN(
        name=name,
        hidden_size=hidden_size,
        use_reflection_pad=use_reflection_pad,
        reflection_bonus=reflection_bonus,
        init_g_ab=spec["init_g_ab"],
        init_g_ba=spec["init_g_ba"],
        structural_multiplier=spec["structural_multiplier"],
        optimization_profile=spec["optimization_profile"],
        model_score_bonus=spec["model_score_bonus"],
    )
