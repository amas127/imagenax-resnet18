import uuid
from dataclasses import dataclass

import nnlogging
import tyro
from flax import nnx
from rich.table import Table
from rich.text import Text

from resnet import (
    eval_step,
    get_eval_dataloader,
    get_optimizer_and_scheduler,
    get_resnet18,
    get_tracer,
    get_train_dataloader,
    train_step,
)

STEPS = int(45e4)


@dataclass
class Argument:
    seed: int
    batch_size: int = 256
    train_workers: int = 16
    eval_workers: int = 16
    log_every: int = 64
    eval_every: int = 10000


def main(args: Argument):
    rngs = nnx.Rngs(args.seed)
    model = get_resnet18(rngs=rngs)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    tracer = get_tracer()
    eval_tracer = get_tracer()
    train_dataloader = get_train_dataloader(
        "data/in1k-train-*.ar",
        nepochs=120,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        worker_count=args.train_workers,
    )
    eval_dataloader = get_eval_dataloader(
        "data/in1k-validation-*.ar",
        batch_size=args.batch_size,
        worker_count=args.eval_workers,
    )
    nnlogging.add_task("train", total=STEPS, description="Training")
    loss = top1acc = top5acc = None
    eval_loss = eval_top1acc = eval_top5acc = None
    for i, batch in enumerate(train_dataloader):
        model.train()
        train_step(model, optimizer, tracer, batch[0], batch[1])
        nnlogging.advance("train", 1)
        if (i + 1) % args.log_every == 0:
            metrics = tracer.compute()
            loss = metrics["loss"].item()
            top1acc = metrics["top1acc"].item() * 100
            top5acc = metrics["top5acc"].item() * 100
            current_lr = scheduler(i)
            nnlogging.debug(
                __name__,
                " ".join(("(step %06d)", "(lr %.1e)", ""))
                + "  |  ".join(("loss: %.4f", "top1acc: %.2f", "top5acc: %.2f")),
                i,
                current_lr,
                loss,
                top1acc,
                top5acc,
            )
            nnlogging.track(
                i,
                metrics={"loss": loss, "top1acc": top1acc, "top5acc": top5acc},
                context={"phase": "train"},
            )
            tracer.reset()
        if (i + 1) % args.eval_every == 0:
            eval_tracer.reset()
            model.eval()
            for batch in eval_dataloader:
                eval_step(model, eval_tracer, batch[0], batch[1])
            metrics = eval_tracer.compute()
            eval_loss = metrics["loss"].item()
            eval_top1acc = metrics["top1acc"].item() * 100
            eval_top5acc = metrics["top5acc"].item() * 100
            table = Table(title="Evaluation")
            table.add_column(Text("Field", justify="center"), style="cyan")
            table.add_column(Text("Value", justify="center"), style="magenta")
            table.add_row("Loss", f"{eval_loss:.4f}")
            table.add_row("Top-1 Accuracy", f"{eval_top1acc:.2f}")
            table.add_row("Top-5 Accuracy", f"{eval_top5acc:.2f}")
            nnlogging.render(__name__, "INFO", table)
            nnlogging.track(
                i,
                metrics={
                    "loss": eval_loss,
                    "top1acc": eval_top1acc,
                    "top5acc": eval_top5acc,
                },
                context={"phase": "evaluation"},
            )
        if (i + 1) >= STEPS:
            break

    nnlogging.add_summaries(
        {
            "train-loss": loss,
            "train-top1acc": top1acc,
            "train-top5acc": top5acc,
            "eval-loss": eval_loss,
            "eval-top1acc": eval_top1acc,
            "eval-top5acc": eval_top5acc,
        }
    )
    nnlogging.update_status("SUCCESSFUL")
    nnlogging.close_run()
    nnlogging.archive_run()


if __name__ == "__main__":
    args = tyro.cli(Argument)
    nnlogging.configure_logger([__name__], level="DEBUG", propagate=False)
    nnlogging.add_branch(
        ("console", "stderr"), ("file", open("main.log", "w+")), logger=__name__
    )
    nnlogging.configure_run(
        uuid=uuid.uuid4(), experiment="ImageNet1K", run=str(args.seed)
    )
    nnlogging.add_hparams(
        {"seed": args.seed, "batch_size": args.batch_size, "steps": STEPS}
    )
    main(args)
