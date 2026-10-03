"""Per-run folders under data/experiments/<variant>/<variant>_<timestamp>/."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

from hale_vlm.config.run import RunConfig


def variant_root(cfg: RunConfig) -> Path:
    return Path(cfg.experiment.root) / cfg.variant


def default_run_name(cfg: RunConfig, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{cfg.variant}_{stamp}"


def latest_run_dir(cfg: RunConfig) -> Path | None:
    link = variant_root(cfg) / "latest"
    if link.exists():
        return link.resolve()
    return None


def _copy_source_yamls(source: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    path = source.resolve()
    while path not in seen:
        seen.add(path)
        if not path.exists():
            break
        (dest_dir / path.name).write_text(path.read_text())
        raw = yaml.safe_load(path.read_text()) or {}
        inherits = raw.get("inherits")
        if not inherits:
            break
        path = (path.parent / inherits).resolve()


def _dump_resolved_config(cfg: RunConfig, run_dir: Path) -> None:
    payload = cfg.model_dump()
    payload.get("experiment", {}).pop("source_yaml", None)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def _bind_paths(cfg: RunConfig, run_dir: Path, *, dump_config: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "viz").mkdir(exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    cfg.train.checkpoint_path = str(run_dir / "checkpoints" / "last.pt")
    cfg.logging.log_file = str(run_dir / "logs" / "train.log")
    cfg.logging.tensorboard_dir = str(run_dir / "tb")
    cfg.viz.output_dir = str(run_dir / "viz")
    cfg.experiment.name = run_dir.name
    if dump_config:
        _dump_resolved_config(cfg, run_dir)
        if cfg.experiment.source_yaml:
            _copy_source_yamls(Path(cfg.experiment.source_yaml), run_dir / "configs")
    logger.info("experiment dir {}", run_dir.resolve())


def _point_latest(cfg: RunConfig, run_dir: Path) -> None:
    root = variant_root(cfg)
    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        logger.warning("could not update latest symlink at {}", latest)


def apply_experiment_layout(
    cfg: RunConfig, *, create: bool = True, name: str | None = None
) -> Path | None:
    """Rewrite checkpoint/log/tb/viz paths onto one experiment folder.

    create=True  — new ``{variant}_{YYYYMMDD_HHMMSS}`` run (or ``name`` if given).
    create=False — attach to ``name`` or ``<variant>/latest``.
    """
    if not cfg.experiment.enabled:
        return None
    if name:
        cfg.experiment.name = name
    root = variant_root(cfg)
    root.mkdir(parents=True, exist_ok=True)

    if cfg.experiment.name:
        run_dir = root / cfg.experiment.name
        if not create and not run_dir.exists():
            raise FileNotFoundError(f"experiment folder not found: {run_dir}")
        _bind_paths(cfg, run_dir, dump_config=create)
        if create:
            _point_latest(cfg, run_dir)
        return run_dir

    if not create:
        run_dir = latest_run_dir(cfg)
        if run_dir is None:
            logger.info("no latest experiment under {}", root)
            return None
        _bind_paths(cfg, run_dir, dump_config=False)
        return run_dir

    if cfg.train.resume:
        run_dir = latest_run_dir(cfg)
        if run_dir is not None:
            _bind_paths(cfg, run_dir, dump_config=True)
            return run_dir

    run_dir = root / default_run_name(cfg)
    _bind_paths(cfg, run_dir, dump_config=True)
    _point_latest(cfg, run_dir)
    return run_dir
