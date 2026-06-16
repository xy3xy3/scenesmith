from omegaconf import OmegaConf

from main import apply_scenebenchmark_critic_toggle


def test_apply_scenebenchmark_critic_toggle_defaults_to_existing_config() -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {
                "scenebenchmark_critic": {
                    "enabled": False,
                }
            }
        }
    )

    apply_scenebenchmark_critic_toggle(cfg)

    assert cfg.experiment.scenebenchmark_critic.enabled is False


def test_apply_scenebenchmark_critic_toggle_overrides_enabled_flag() -> None:
    cfg = OmegaConf.create(
        {
            "scenebenchmark_critic_enabled": "true",
            "experiment": {
                "scenebenchmark_critic": {
                    "enabled": False,
                }
            },
        }
    )

    apply_scenebenchmark_critic_toggle(cfg)

    assert cfg.experiment.scenebenchmark_critic.enabled is True


def test_apply_scenebenchmark_critic_toggle_accepts_false_strings() -> None:
    cfg = OmegaConf.create(
        {
            "scenebenchmark_critic_enabled": "off",
            "experiment": {
                "scenebenchmark_critic": {
                    "enabled": True,
                }
            },
        }
    )

    apply_scenebenchmark_critic_toggle(cfg)

    assert cfg.experiment.scenebenchmark_critic.enabled is False
