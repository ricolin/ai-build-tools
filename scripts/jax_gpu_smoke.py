#!/usr/bin/env python3
import json
import sys


def main() -> int:
    import jax
    import jax.numpy as jnp

    devices = [str(device) for device in jax.devices()]
    result = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": devices,
    }
    if jax.default_backend() != "gpu" or not devices:
        result["error"] = "JAX did not select the GPU backend"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    @jax.jit
    def calculate(value):
        return jnp.sum(value @ value.T)

    value = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    result["jax_sum"] = float(calculate(value).block_until_ready())
    if result["jax_sum"] != 3680.0:
        result["error"] = "unexpected deterministic JAX result"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
