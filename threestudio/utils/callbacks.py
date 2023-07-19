import os
import shutil
import subprocess

import pytorch_lightning

from threestudio.utils.config import dump_config
from threestudio.utils.misc import parse_version

if parse_version(pytorch_lightning.__version__) > parse_version("1.8"):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(
            self.save_root,
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version}",
        )


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)

    def get_file_list(self):
        return [
            b.decode()
            for b in set(
                subprocess.check_output(
                    'git ls-files -- ":!:load/*"', shell=True
                ).splitlines()
            )
            | set(  # hard code, TODO: use config to exclude folders or files
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


class ConfigSnapshotCallback(VersionedCallback):
    def __init__(self, config_path, config, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
        self.config_path = config_path
        self.config = config

    @rank_zero_only
    def save_config_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        dump_config(os.path.join(self.savedir, "parsed.yaml"), self.config)
        shutil.copyfile(self.config_path, os.path.join(self.savedir, "raw.yaml"))

    def on_fit_start(self, trainer, pl_module):
        self.save_config_snapshot()


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class ProgressCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self._file_handle = None

    @property
    def file_handle(self):
        if self._file_handle is None:
            self._file_handle = open(self.save_path, "w")
        return self._file_handle

    @rank_zero_only
    def write(self, msg: str) -> None:
        self.file_handle.seek(0)
        self.file_handle.truncate()
        self.file_handle.write(msg)
        self.file_handle.flush()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.write(
            f"Generation progress: {pl_module.true_global_step / trainer.max_steps * 100:.2f}%"
        )

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        self.write(f"Rendering validation image ...")

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        self.write(f"Rendering video ...")

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        self.write(f"Exporting mesh assets ...")
