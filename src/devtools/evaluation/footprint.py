"""Account for declared model-store aliases in an observed footprint."""

import stat
from pathlib import Path

from devtools.evaluation.resource_budget import daily_scope_complete


def model_aliases(tree, aliases, model_store, snapshot, footprint, roots, models):
    accounted, unresolved = {}, []
    try:
        store = models.default_store_root() if model_store is None else model_store
        packs = (
            {
                models.pack_directory(store, pack).resolve()
                for group in models.RECIPE_PACKS.values()
                for pack in group
            }
            if isinstance(store, Path)
            else set()
        )
    except (OSError, ValueError):
        packs = set()
    for alias in aliases:
        try:
            path = Path(alias)
            parts = path.relative_to(tree / "workspaces").parts
            target = path.resolve(strict=True)
            if (
                len(parts) != 4
                or parts[1] != "evidence-cro-authority"
                or not parts[2].startswith("semantic-model")
                or not target.is_dir()
                or not any(target.is_relative_to(pack) for pack in packs)
            ):
                raise ValueError("unregistered model link")
            name = f"model-{len(accounted)}"
            identity = target.stat()
            files = snapshot(tree, root=target, aliases=unresolved)
            final = target.stat()
            if (identity.st_dev, identity.st_ino) != (final.st_dev, final.st_ino) or path.resolve(
                strict=True
            ) != target:
                raise ValueError("model link changed during snapshot")
            footprint.update({name + "/" + key: value for key, value in files.items()})
            roots[name] = target
            accounted[alias] = str(target)
        except (OSError, ValueError):
            unresolved.append(alias)
    return accounted, unresolved


def workspace_root(tree):
    """Identify the observed workspace directory without accepting a missing or linked root."""
    root = tree / "workspaces"
    try:
        info = root.lstat()
        if (
            root.is_symlink()
            or root.is_junction()
            or not stat.S_ISDIR(info.st_mode)
            or not info.st_ino
        ):
            return None
        return [info.st_dev, info.st_ino]
    except OSError:
        return None


def workspace_before(tree, model_store, snapshot, models):
    """Capture workspace and its admitted physical aliases before the daily phase."""
    aliases, physical, roots = [], {}, {}
    root = workspace_root(tree)
    files = snapshot(tree, aliases=aliases)
    accounted, unresolved = model_aliases(
        tree, aliases, model_store, snapshot, physical, roots, models
    )
    if files.keys() & physical.keys():
        raise ValueError("budget.invalid_inode_receipt: workspace model path collision")
    files.update(physical)
    root = root if root is not None and root == workspace_root(tree) else None
    return files, {
        "complete": not unresolved and root is not None,
        "workspace_root": root,
        "accounted_aliases": accounted,
        "excluded_aliases": unresolved,
    }


def workspace_after(files, physical):
    """Add the same physical model roots used by the footprint to workspace growth."""
    models = {key: value for key, value in physical.items() if key.startswith("model-")}
    if files.keys() & models.keys():
        raise ValueError("budget.invalid_inode_receipt: workspace model path collision")
    return files | models


def workspace_scope_after(tree, identity, roots, unresolved, accounted):
    """Record whether the workspace root remained stable throughout the final walk."""
    identity = identity if identity is not None and identity == workspace_root(tree) else None
    return {
        "complete": not unresolved and identity is not None,
        "workspace_root": identity,
        "roots": {name: str(path) for name, path in roots.items()},
        "excluded_aliases": unresolved,
        "accounted_aliases": accounted,
    }


def workspace_totals(before, after, scope_before, scope_after, measure):
    """Do not publish empty numeric measurements when either workspace scope is unproved."""
    if daily_scope_complete(scope_before, scope_after):
        return measure(before, after)
    return dict(
        before_bytes=None,
        after_bytes=None,
        by_writer=None,
        reason="budget.daily_scope_incomplete_or_changed",
    )
