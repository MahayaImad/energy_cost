"""ClientApp: local training (optionally DP-SGD) on a Dirichlet partition."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from energyfl import dp
from energyfl.task import (
    Net,
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
    )
    return trainloader, valloader


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    epsilon = cfg["epsilon"]

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    trainloader, _ = _setup(context)
    n = len(trainloader.dataset)
    lr = float(msg.content["config"]["lr"])
    epochs = int(cfg["local-epochs"])

    metrics = {"num-examples": n}

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
        train_loss, state_dict, nsteps = dp.train_private(
            model, trainloader, epochs, lr, device, noise_multiplier=sigma
        )
        # sigma varies with partition size; both belong in the paper's
        # reproducibility table.
        metrics["noise_multiplier"] = sigma
        metrics["clipping_norm"] = dp.CLIPPING_NORM
    else:
        model.to(device)
        train_loss = train_fn(model, trainloader, epochs, lr, device)
        state_dict = model.state_dict()
        nsteps = dp.steps_per_epoch(n, int(cfg["batch-size"])) * epochs

    # Must be identical across every epsilon, or the energy axis is measuring
    # workload size instead of the cost of privacy. Check this first.
    metrics["local_steps"] = nsteps
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
    model = Net()
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
    metrics["expected_steps"] = dp.steps_per_epoch(n, int(cfg["batch-size"])) * epochs
    return Message(content=content, reply_to=msg)