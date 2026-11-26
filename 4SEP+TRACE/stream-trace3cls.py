import argparse
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from queue import Full
import shutil
import struct
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

import arithmeticcoding_fast as arithmetic
from FETRACE3 import SLiMPerformer


MAGIC = b"SEPTRACE\x01"
MODEL_DEFAULTS = {
    "batch_size": 512, "seq_len": 64, "vocab_size": 256, "vocab_dim": 64,
    "hidden_dim": 256, "n_layers": 1, "ffn_dim": 4096, "n_heads": 8,
    "feature_type": "sqr", "compute_type": "iter", "learning_rate": 1e-3,
    "weight_decay": 0.0, "random_seed": 0,
}
UNIFORM_CDF = np.empty(257, dtype=np.uint64)
UNIFORM_CDF[0] = 0
UNIFORM_CDF[1:] = np.cumsum(np.full(256, 10_000_000 / 256 + 1))
MAX_HEADER = 65536


def split_input(size, batch_size, history):
    # Preserve the benchmark's shard boundaries. All remaining bytes belong to
    # the tail, even when the number of sliding windows divides evenly.
    shard_size = (size - history) // batch_size
    if shard_size <= history:
        return 0, size
    return shard_size, size - batch_size * shard_size


def frequencies(probabilities):
    result = np.zeros((probabilities.shape[0], 257), dtype=np.uint64)
    result[:, 1:] = np.cumsum(probabilities * 10_000_000 + 1, axis=1)
    return result


def init_model(config, device):
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Older PyTorch CUDA builds report cumsum as unsupported in strict mode.
    # Enable deterministic kernels where available and verify decoded bytes.
    torch.use_deterministic_algorithms(True, warn_only=device.type == "cuda")
    torch.manual_seed(config["random_seed"])
    model = SLiMPerformer(
        config["vocab_size"], config["vocab_dim"], config["hidden_dim"],
        config["n_layers"], config["ffn_dim"], config["n_heads"],
        config["feature_type"], config["compute_type"],
        batch_size=config["batch_size"], seq_len=config["seq_len"], device=device,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    return model, optimizer


def update_model(logits, labels, optimizer):
    F.cross_entropy(logits, labels).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def make_encoder(file):
    return arithmetic.ArithmeticEncoder(32, arithmetic.BitOutputStream(file))


def make_decoder(file):
    return arithmetic.ArithmeticDecoder(32, arithmetic.BitInputStream(file))


def write_uniform(encoder, symbols):
    for symbol in symbols:
        encoder.write(UNIFORM_CDF, int(symbol))


def read_uniform(decoder, count):
    return np.fromiter(
        (decoder.read(UNIFORM_CDF, 256) for _ in range(count)), dtype=np.uint8, count=count,
    )


def write_shards(queue, paths, prefixes):
    with ExitStack() as stack:
        encoders = [make_encoder(stack.enter_context(path.open("wb"))) for path in paths]
        for encoder, prefix in zip(encoders, prefixes):
            write_uniform(encoder, prefix)
        while (batch := queue.get()) is not None:
            cdfs, labels = batch
            for encoder, cdf, label in zip(encoders, cdfs, labels):
                encoder.write(cdf, int(label))
        for encoder in encoders:
            encoder.finish()
            encoder.output.close()


def put_batch(queue, writer, batch):
    while writer.is_alive():
        try:
            queue.put(batch, timeout=0.25)
            return
        except Full:
            continue
    raise RuntimeError(f"Arithmetic encoder process exited with code {writer.exitcode}")


def cpu_predictions(windows, model, optimizer, device):
    for window in windows:
        batch = torch.from_numpy(window.copy()).to(device=device, dtype=torch.long)
        logits = model(batch[:, :-1].contiguous())[:, -1, :]
        probabilities = F.softmax(logits.detach(), dim=1).numpy()
        update_model(logits, batch[:, -1].contiguous(), optimizer)
        yield probabilities, window[:, -1].copy()


def cuda_predictions(windows, model, optimizer, device):
    # Model updates share one stream so the decoder can replay them in order.
    compute = torch.cuda.current_stream(device)
    upload = torch.cuda.Stream(device=device)
    download = torch.cuda.Stream(device=device)
    pending = None

    def prefetch(window):
        if window is None:
            return None
        host = torch.from_numpy(window.copy()).pin_memory()
        with torch.cuda.stream(upload):
            batch = host.to(device, non_blocking=True)
            ready = upload.record_event()
        return batch, ready, window[:, -1].copy()

    windows = iter(windows)
    current = prefetch(next(windows, None))
    while current is not None:
        batch, uploaded, labels = current
        compute.wait_event(uploaded)
        batch.record_stream(compute)
        batch = batch.long()
        logits = model(batch[:, :-1].contiguous())[:, -1, :]
        probabilities = F.softmax(logits.detach(), dim=1)
        predicted = compute.record_event()
        with torch.cuda.stream(download):
            download.wait_event(predicted)
            host_probabilities = probabilities.to("cpu", non_blocking=True)
            probabilities.record_stream(download)
            downloaded = download.record_event()

        current = prefetch(next(windows, None))
        update_model(logits, batch[:, -1].contiguous(), optimizer)

        # Queue the next round's GPU work before waiting on the previous copy.
        if pending is not None:
            host, ready, previous_labels = pending
            ready.synchronize()
            yield host.numpy(), previous_labels
        pending = host_probabilities, downloaded, labels

    if pending is not None:
        host, ready, labels = pending
        ready.synchronize()
        yield host.numpy(), labels
    compute.synchronize()


def encode_shards(shards, history, paths, config, device, print_step):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=8)
    writer = ctx.Process(target=write_shards, args=(queue, paths, shards[:, :history].copy()))
    writer.start()
    try:
        model, optimizer = init_model(config, device)
        windows = (shards[:, i:i + history + 1] for i in range(shards.shape[1] - history))
        predict = cuda_predictions if device.type == "cuda" else cpu_predictions
        for index, (probabilities, labels) in enumerate(predict(windows, model, optimizer, device)):
            put_batch(queue, writer, (frequencies(probabilities), labels))
            if print_step and (index + 1) % print_step == 0:
                print(f"Encoded {index + 1} prediction batches", flush=True)
        put_batch(queue, writer, None)
        writer.join()
        if writer.exitcode:
            raise RuntimeError(f"Arithmetic encoder process exited with code {writer.exitcode}")
    finally:
        if writer.is_alive():
            writer.terminate()
            writer.join()
        if writer.exitcode != 0:
            queue.cancel_join_thread()
        queue.close()
        queue.join_thread()


def read_exact(source, length):
    data = source.read(length)
    if len(data) != length:
        raise ValueError("Truncated SEP+TRACE archive")
    return data


def pack_part(output, path):
    with path.open("rb") as source:
        output.write(struct.pack(">Q", path.stat().st_size))
        shutil.copyfileobj(source, output)


def unpack_part(source, path):
    remaining, = struct.unpack(">Q", read_exact(source, 8))
    if not remaining:
        raise ValueError("Empty arithmetic coding stream")
    with path.open("wb") as output:
        while remaining:
            block = read_exact(source, min(remaining, 1024 * 1024))
            output.write(block)
            remaining -= len(block)


def compress(source, destination, config, device, print_step):
    data = np.frombuffer(source.read_bytes(), dtype=np.uint8)
    history = config["seq_len"]
    batch_size = config["batch_size"]
    shard_size, tail_size = split_input(len(data), batch_size, history)
    split = shard_size * batch_size
    header = json.dumps({
        "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "model": config, "device": device.type,
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
    }).encode("ascii")

    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        temp = Path(temp)
        paths = [temp / str(i) for i in range(batch_size)] if shard_size else []
        if paths:
            encode_shards(data[:split].reshape(batch_size, shard_size), history, paths, config, device, print_step)
        if tail_size:
            tail = temp / "tail"
            with tail.open("wb") as part:
                encoder = make_encoder(part)
                write_uniform(encoder, data[split:])
                encoder.finish()
                encoder.output.close()
            paths.append(tail)
        archive = temp / "archive"
        with archive.open("wb") as output:
            output.write(MAGIC)
            output.write(struct.pack(">I", len(header)))
            output.write(header)
            for path in paths:
                pack_part(output, path)
        os.replace(archive, destination)
    return len(data)


def decode_shards(paths, shards, config, device):
    history = config["seq_len"]
    model, optimizer = init_model(config, device)
    with ExitStack() as stack:
        decoders = [make_decoder(stack.enter_context(path.open("rb"))) for path in paths]
        for decoder, row in zip(decoders, shards):
            row[:history] = read_uniform(decoder, history)
        for index in range(shards.shape[1] - history):
            target = index + history
            batch = torch.from_numpy(shards[:, index:target].copy()).to(device=device, dtype=torch.long)
            logits = model(batch)[:, -1, :]
            cdfs = frequencies(F.softmax(logits.detach(), dim=1).cpu().numpy())
            labels = np.fromiter(
                (decoder.read(cdf, 256) for decoder, cdf in zip(decoders, cdfs)),
                dtype=np.uint8, count=len(decoders),
            )
            shards[:, target] = labels
            update_model(logits, torch.from_numpy(labels).to(device=device, dtype=torch.long), optimizer)


def decompress(source, destination, device):
    with source.open("rb") as archive, tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        if archive.read(len(MAGIC)) != MAGIC:
            raise ValueError("Unsupported archive format. This decoder requires a versioned SEP+TRACE archive.")
        length, = struct.unpack(">I", read_exact(archive, 4))
        if length > MAX_HEADER:
            raise ValueError("Invalid SEP+TRACE header length")
        header = json.loads(read_exact(archive, length))
        config = header["model"]
        validate_config(config)
        if type(header["size"]) is not int or header["size"] < 0:
            raise ValueError("Invalid original size")
        size = header["size"]
        batch_size = config["batch_size"]
        shard_size, tail_size = split_input(size, batch_size, config["seq_len"])
        if shard_size and header["device"] != device.type:
            raise ValueError(f"Decode this archive with --device={header['device']} to replay the model")
        if shard_size and device.type == "cpu":
            torch.set_num_threads(header["cpu_threads"])

        temp = Path(temp)
        paths = [temp / str(i) for i in range(batch_size)] if shard_size else []
        tail = temp / "tail"
        for path in paths + ([tail] if tail_size else []):
            unpack_part(archive, path)
        if archive.read(1):
            raise ValueError("Unexpected data after the last SEP+TRACE stream")

        data = np.empty(size, dtype=np.uint8)
        split = batch_size * shard_size
        if paths:
            decode_shards(paths, data[:split].reshape(batch_size, shard_size), config, device)
        if tail_size:
            with tail.open("rb") as part:
                data[split:] = read_uniform(make_decoder(part), tail_size)
        if hashlib.sha256(data).hexdigest() != header["sha256"]:
            raise ValueError("Decoded bytes failed SHA-256 verification")
        restored = temp / "restored"
        with restored.open("wb") as output:
            data.tofile(output)
        os.replace(restored, destination)
    return size


def validate_config(config):
    if set(config) != set(MODEL_DEFAULTS):
        raise ValueError("Invalid SEP+TRACE model settings")
    if config["vocab_size"] != 256 or config["vocab_dim"] != 64 or config["hidden_dim"] != 256:
        raise ValueError("SEP+TRACE uses vocab_size=256, vocab_dim=64 and hidden_dim=256")
    if config["seq_len"] < 4 or config["batch_size"] < 1:
        raise ValueError("seq_len must be at least 4 and batch_size must be positive")
    if config["n_heads"] < 1 or 256 % config["n_heads"]:
        raise ValueError("n_heads must divide 256")
    if config["n_layers"] < 1 or config["ffn_dim"] < 1:
        raise ValueError("n_layers and ffn_dim must be positive")
    if config["compute_type"] not in {"iter", "ps", "parallel_ps"}:
        raise ValueError("compute_type must be iter, ps or parallel_ps")
    feature_type = config["feature_type"]
    base, _, factor = feature_type.partition("_")
    if feature_type not in {"relu", "elu+1", "sqr", "abs", "favor+"} and not (
        base == "favor+" and factor.isdecimal() and int(factor) > 0
    ):
        raise ValueError("Invalid attention feature_type")


def main():
    parser = argparse.ArgumentParser(description="Compress or decompress a binary file with SEP+TRACE")
    parser.add_argument("--mode", choices=("compress", "decompress"), default="compress")
    parser.add_argument("--input_dir", type=Path, default=Path("../data/book1"), help="Input file")
    parser.add_argument("--output", type=Path, help="Destination file")
    parser.add_argument("--prefix", default="_book1", help="Prefix for the default archive filename")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--print_step", type=int, default=1000, help="Progress interval, 0 disables")
    for name, default in MODEL_DEFAULTS.items():
        parser.add_argument(f"--{name}", type=type(default), default=default)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("No CUDA device available. Use --device=cpu.")
    device = torch.device(args.device)
    config = {name: getattr(args, name) for name in MODEL_DEFAULTS}
    if args.mode == "compress":
        dest = args.output or Path(
            f"{args.prefix}_{args.vocab_dim}_{args.hidden_dim}_{args.ffn_dim}_"
            f"bs{args.batch_size}_{args.n_layers}_seq{args.seq_len}.compressed.combined"
        )
    else:
        dest = args.output or Path(str(args.input_dir) + ".restored")
    if args.input_dir.resolve() == dest.resolve():
        parser.error("Input and output must be different files")
    start = time.perf_counter()
    try:
        if args.mode == "compress":
            validate_config(config)
            size = compress(args.input_dir, dest, config, device, args.print_step)
            print(f"Compressed {size} bytes to {dest.stat().st_size} bytes in {time.perf_counter() - start:.3f}s")
        else:
            size = decompress(args.input_dir, dest, device)
            print(f"Decompressed {size} bytes in {time.perf_counter() - start:.3f}s (SHA-256 verified)")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"SEP+TRACE: {error}\n")


if __name__ == "__main__":
    main()
