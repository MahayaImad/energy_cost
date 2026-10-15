"""ClientApp: one round of local training, private or not."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from energyfl import dp
from energyfl.task import (build_model, get_device, load_partition, set_seed,
                           test, train)

app = ClientApp()


def _setup(msg: Message, context: Context):
    cfg = context.run_config
    pid = context.node_config["partition-id"]
    set_seed(int(cfg["seed"]) * 1000 + pid)
    model = build_model(cfg["dataset"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    loaders = load_partition(pid, context.node_config["num-partitions"],
                             int(cfg["batch-size"]),
                             float(cfg["dirichlet-alpha"]), int(cfg["seed"]),
                             cfg["dataset"])
    return cfg, pid, model, loaders


@app.train()
def train_fn(msg: Message, context: Context):
    cfg, pid, model, (trainloader, _) = _setup(msg, context)
    device = get_device()
    epochs = int(cfg["local-epochs"])
    lr = float(cfg["learning-rate"])
    metrics = {"num-examples": len(trainloader.dataset), "partition_id": pid}

    if dp.is_private(cfg["epsilon"]):
        # sigma was computed by the server before any measurement started.
        sigma = float(msg.content["config"]["sigmas"][pid])
        state = dp.train_private(model, trainloader, epochs, lr, device,
                                 sigma, float(cfg["clipping-norm"]))
        metrics["noise_multiplier"] = sigma
    else:
        train(model, trainloader, epochs, lr, device)
        state = model.state_dict()

    return Message(RecordDict({"arrays": ArrayRecord(state),
                               "metrics": MetricRecord(metrics)}),
                   reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    _, _, model, (_, valloader) = _setup(msg, context)
    loss, acc = test(model, valloader, get_device())
    return Message(RecordDict({"metrics": MetricRecord({
        "eval_loss": loss, "eval_acc": acc,
        "num-examples": len(valloader.dataset)})}), reply_to=msg)
