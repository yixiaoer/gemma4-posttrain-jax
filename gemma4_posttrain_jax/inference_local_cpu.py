"""固定 tpu-inference 加载器的进程内适配：非零 rank 使用本地 CPU。"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from typing import Any

LOADERS = (
    (
        "tpu_inference.layers.common.utils",
        "cpu_mesh",
        "fd7ed4aa302fdf8c56f0b2531da6e292fa7fbf973e6eb8bc23b0900fd92df384",
    ),
    (
        "tpu_inference.models.jax.utils.weight_utils",
        "model_weights_single_file_generator",
        "fa429aba3d765a7ceb55f8d4c21079e9140466ed15891e0059c4d2f6c5ef859d",
    ),
)


class LocalCpuLoaders:
    """退出或安装中途失败时恢复原函数及缓存；不修改安装目录。"""

    def __init__(self) -> None:
        self.originals: list[tuple[Any, str, Any]] = []
        self.metadata: list[dict[str, str]] = []

    def __enter__(self) -> LocalCpuLoaders:
        if self.originals:
            raise RuntimeError("CPU 加载器适配不允许重入")
        try:
            for module_name, name, expected in LOADERS:
                module: Any = importlib.import_module(module_name)
                function = getattr(module, name)
                source = inspect.getsource(function)
                digest = hashlib.sha256(source.encode()).hexdigest()
                if digest != expected or source.count('jax.devices("cpu")') != 1:
                    raise RuntimeError(f"CPU 加载器不符合固定版本：{module_name}.{name}")
                candidate = source.replace('jax.devices("cpu")', 'jax.local_devices(backend="cpu")')
                namespace: dict[str, Any] = {}
                exec(compile(candidate, "<process-local-cpu-loader>", "exec"), function.__globals__, namespace)
                self.originals.append((module, name, function))
                setattr(module, name, namespace[name])
                if name == "cpu_mesh":
                    self.originals.append((module, "_cpu_mesh", module._cpu_mesh))
                    module._cpu_mesh = None
                self.metadata.append({"function": module_name + "." + name, "source_sha256": digest})
            return self
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)
        self.originals.clear()

    def __exit__(self, *args: Any) -> None:
        self.restore()
