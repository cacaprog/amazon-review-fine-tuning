import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class WandbMetricsCallback(TrainerCallback):
    """Log train and eval metrics to W&B at each epoch end."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ):
        if metrics is None:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"epoch": state.epoch, **metrics}, step=state.global_step)
        except ImportError:
            pass
