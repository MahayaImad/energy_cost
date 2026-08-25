"""ClientApp: local training (optionally DP-SGD) on a Dirichlet partition."""

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


def _setup(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = context.run_config

    set_seed(int(cfg["seed"]) * 1000 + partition_id)

    trainloader, valloader = load_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=int(cfg["batch-size"]),
        alpha=float(cfg["dirichlet-alpha"]),
        seed=int(cfg["seed"]),
        dataset=str(cfg.get("dataset", "cifar10")),
        har_split=str(cfg.get("har-split", "official")),
    )
    return trainloader, valloader


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    epsilon = cfg["epsilon"]

    model = build_model(str(cfg.get("dataset", "cifar10")))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    trainloader, _ = _setup(context)
    n = len(trainloader.dataset)
    lr = float(msg.content["config"]["lr"])
    epochs = int(cfg["local-epochs"])

    # partition_id travels with the metrics so the server can record which
    # clients a round actually drew. Without it, a per-round energy difference
    # between two epsilon conditions cannot be told apart from the two rounds
    # simply having sampled differently-sized partitions.
    metrics = {
        "num-examples": n,
        "partition_id": context.node_config["partition-id"],
        # Reported so the server can prove the clients trained on the GPU
        # NVML is watching. If Ray hands a ClientApp no GPU, training falls
        # back to CPU, the run completes normally, and every joule recorded
        # is the energy of an idle card. Silent, and fatal to the results.
        "device_cuda": 1.0 if device.type == "cuda" else 0.0,
    }

    if dp.is_private(epsilon):
        model = dp.validate_model(model)
        sigma = dp.noise_multiplier_for(
            epsilon=float(epsilon),
            n_examples=n,
            batch_size=int(cfg["batch-size"]),
            num_rounds=int(cfg["num-server-rounds"]),
            fraction_train=float(cfg["fraction-train"]),
            local_epochs=epochs,
        )
        # C is fixed at 1.0 for the main sweep -- tuning it per epsilon would
        # confound the privacy-utility comparison. It is configurable only so
        # the C ablation can vary it deliberately, one factor at a time.
        clip = float(cfg.get("clipping-norm", dp.CLIPPING_NORM))
        train_loss, state_dict, nsteps = dp.train_private(
            model, trainloader, epochs, lr, device,
            noise_multiplier=sigma, max_grad_norm=clip,
        )
        # sigma varies with partition size; both belong in the paper's
        # reproducibility table.
        metrics["noise_multiplier"] = sigma
        metrics["clipping_norm"] = clip
    else:
        model.to(device)
        train_loss = train_fn(model, trainloader, epochs, lr, device)
        state_dict = model.state_dict()
        nsteps = dp.steps_per_epoch(n, int(cfg["batch-size"])) * epochs

    # Must be identical across every epsilon, or the energy axis is measuring
    # workload size instead of the cost of privacy. Check this first.
    #
    # expected_steps is the deterministic cap ceil(n/B)*epochs that the
    # accountant was calibrated against; local_steps is what actually ran.
    # Any gap between them means the privacy accounting and the executed
    # workload disagree, so both are logged and analyze.py compares them.
    metrics["local_steps"] = nsteps
    metrics["expected_steps"] = dp.steps_per_epoch(n, int(cfg["batch-size"])) * epochs
    metrics["train_loss"] = train_loss

    content = RecordDict(
        {
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Evaluation is never privatised: it reads the global model, not local
    # gradients, and consumes no privacy budget.
    model = build_model(str(context.run_config.get("dataset", "cifar10")))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    _, valloader = _setup(context)
    loss, acc = test_fn(model, valloader, device)

    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": loss,
                    "eval_acc": acc,
                    "num-examples": len(valloader.dataset),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)