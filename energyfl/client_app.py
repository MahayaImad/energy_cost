"""ClientApp: one round of local training, with or without DP-SGD."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from energyfl import dp
from energyfl.task import (
    build_model,
    get_device,
    load_partition,
    set_seed,
    test_fn,
    train_fn,
)

app = ClientApp()


def _dataloaders(context: Context):
    partition_id = context.node_config["partition-id"]
    cfg = context.run_config
    set_seed(int(cfg["seed"]) * 1000 + partition_id)
    return load_partition(
        partition_id=partition_id,
        num_partitions=context.node_config["num-partitions"],
        batch_size=int(cfg["batch-size"]),
        alpha=float(cfg["dirichlet-alpha"]),
        seed=int(cfg["seed"]),
        dataset=str(cfg["dataset"]),
    )


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    epsilon = cfg["epsilon"]
    batch_size = int(cfg["batch-size"])
    epochs = int(cfg["local-epochs"])

    model = build_model(str(cfg["dataset"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    trainloader, _ = _dataloaders(context)
    n = len(trainloader.dataset)
    lr = float(msg.content["config"]["lr"])

    metrics = {
        "num-examples": n,
        # Which clients a round actually drew. Without it, a per-round energy
        # difference between two epsilon conditions cannot be told apart from
        # the two rounds having sampled differently-sized partitions.
        "partition_id": context.node_config["partition-id"],
        # Proof the client trained on the GPU that NVML is watching. If Ray
        # hands a ClientApp no GPU, training falls back to CPU, the run
        # completes normally, and every joule recorded is an idle card.
        "device_cuda": 1.0 if device.type == "cuda" else 0.0,
        # What ran, against what the accountant was calibrated for. A gap
        # between them means the privacy claim and the workload disagree.
        "expected_steps": dp.steps_per_epoch(n, batch_size) * epochs,
    }

    if dp.is_private(epsilon):
        model = dp.validate_model(model)
        sigma = dp.noise_multiplier_for(
            epsilon=float(epsilon),
            n_examples=n,
            batch_size=batch_size,
            num_rounds=int(cfg["num-server-rounds"]),
            fraction_train=float(cfg["fraction-train"]),
            local_epochs=epochs,
        )
        # C is fixed for the main sweep; it is configurable only so the
        # clipping ablation can vary it, one factor at a time.
        clip = float(cfg["clipping-norm"])
        train_loss, state_dict, nsteps = dp.train_private(
            model, trainloader, epochs, lr, device,
            noise_multiplier=sigma, max_grad_norm=clip,
        )
        # sigma varies with partition size; the range goes in the paper.
        metrics["noise_multiplier"] = sigma
        metrics["clipping_norm"] = clip
    else:
        model.to(device)
        train_loss = train_fn(model, trainloader, epochs, lr, device)
        state_dict = model.state_dict()
        nsteps = metrics["expected_steps"]

    metrics["local_steps"] = nsteps
    metrics["train_loss"] = train_loss

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Never privatised: this reads the global model, not local gradients, and
    # consumes no privacy budget.
    model = build_model(str(context.run_config["dataset"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    _, valloader = _dataloaders(context)
    loss, acc = test_fn(model, valloader, device)

    return Message(
        content=RecordDict({
            "metrics": MetricRecord({
                "eval_loss": loss,
                "eval_acc": acc,
                "num-examples": len(valloader.dataset),
            })
        }),
        reply_to=msg,
    )
