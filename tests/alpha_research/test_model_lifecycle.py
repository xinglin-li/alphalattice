"""Declared calendars, causal training, and immutable model renewal boundaries."""

import os
import pickle
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Barrier, Lock, get_ident
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
    prepare_frozen_price_volume_features,
    prepare_frozen_price_volume_history,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AlphaModelLifecycleAdmission,
    AlphaRefitBudget,
    admit_component_inference,
    fit_alpha_refit_child,
    prepare_alpha_refit,
    publish_training_observations,
    verified_lifecycle_admissions,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    DEFAULT_MODEL_LIFECYCLE,
    AlphaModelLifecycleRecipe,
    model_lifecycle_of,
    resolve_alpha_refit_plan,
)
from alphalattice.investment.alpha_research.scores.product_replay import live_vintages
from alphalattice.kernel.shared_kernel.identity import canonical_hash

COMPONENT = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component("G6_R0_FAST_REBOUND")


@pytest.mark.parametrize("interval", (1, 3, 12))
def test_declared_periods_resolve_training_without_changing_the_executor(interval):
    frozen = AlphaModelLifecycleRecipe.from_component(COMPONENT)
    values = frozen.model_dump(exclude={"content_hash"})
    values.update(month_interval=interval, training_window_sessions=20)
    rule = AlphaModelLifecycleRecipe.create(**values)
    days = tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(600))
    vintage = rule.vintages(date(2026, 7, 2))[0]
    plan = resolve_alpha_refit_plan(
        lifecycle=rule,
        vintage=vintage,
        sessions=days,
        component_recipe_hash=COMPONENT.recipe_hash,
        source_binding_hash="a" * 64,
        ordered_listing_ids=("a", "b"),
        ordered_feature_ids=COMPONENT.ordered_feature_ids,
    )
    assert len(plan.training_sessions) == 20
    assert len(plan.purge_sessions) == 5
    assert plan.training_sessions[-1] < plan.purge_sessions[0] < plan.first_formation
    assert frozen.vintages(date(2026, 7, 2)) == live_vintages(date(2026, 7, 2), count=4)
    assert rule.content_hash != frozen.content_hash
    with pytest.raises(ValueError, match="identity_invalid"):
        AlphaModelLifecycleRecipe.model_validate({**rule.model_dump(), "purge_sessions": 0})


def _source(day_count=250):
    rng = np.random.default_rng(1729)
    days = tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(day_count))
    names = tuple(f"listing-{i}" for i in range(8))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), len(names))), axis=0))
    return FrozenPriceVolumeInputs(
        formation_sessions=days,
        ordered_listing_ids=names,
        sector_by_listing_id={name: "sector" for name in names},
        open=close * 0.995,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=rng.uniform(1000, 2000, close.shape),
        market_context_values=rng.normal(size=(len(days), 3)),
        sector_trend_values=rng.normal(size=(len(days), 1)),
        source_binding_hash="b" * 64,
    )


@pytest.fixture
def retained_lifecycle_admission(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path / "runtime")
    source = _source(380)
    snapshot = store.publish_frozen_observations(source, disposition="SYNTHETIC_INPUT_QA")
    args = dict(
        observation_hash=snapshot.snapshot_hash,
        target_method_id=COMPONENT.target_recipe,
        label_available_sessions=tuple(
            day + timedelta(days=1) for day in source.formation_sessions
        ),
        targets=np.log(source.close / source.open),
        eligible=np.ones(source.close.shape, dtype=np.bool_),
    )
    old = publish_training_observations(store, **args, target_authority_hash="c" * 64)
    current = publish_training_observations(store, **args, target_authority_hash="d" * 64)
    assert old.content_hash != current.content_hash
    assert old.array_file_hash == current.array_file_hash
    rule = AlphaModelLifecycleRecipe.create(
        **{
            **AlphaModelLifecycleRecipe.from_component(COMPONENT).model_dump(
                exclude={"content_hash"}
            ),
            "training_window_sessions": 50,
            "vintage_weights": (1,),
            "seeds": (1729,),
        }
    )
    prepared = tuple(
        prepare_alpha_refit(
            store,
            plan=resolve_alpha_refit_plan(
                lifecycle=rule,
                vintage=vintage,
                sessions=source.formation_sessions,
                component_recipe_hash=COMPONENT.recipe_hash,
                source_binding_hash=observations.content_hash,
                ordered_listing_ids=source.ordered_listing_ids,
                ordered_feature_ids=COMPONENT.ordered_feature_ids,
            ),
            observations=observations,
            component=COMPONENT,
        )
        for vintage, observations in (("2025-07", old), ("2025-10", current))
    )
    authority = AlphaModelLifecycleAdmission.create(
        component=COMPONENT,
        lifecycle=rule,
        observations_hash=current.content_hash,
        prepared=prepared,
        initial_children=(),
        formation_start=date(2025, 10, 1),
        formation_end=date(2025, 10, 1),
        maximum_fit_attempts=0,
        environment_hash="e" * 64,
    )
    root = tmp_path / "admission"
    root.mkdir()
    (root / "model-lifecycle-admission.json").write_text(
        authority.model_dump_json(), encoding="utf-8", newline="\n"
    )
    return store, root, authority, (old, current)


def test_admission_verifies_unique_arrays_in_parallel_after_serial_bindings(
    retained_lifecycle_admission, monkeypatch
):

    store, root, authority, observations = retained_lifecycle_admission
    snapshot = store.load_frozen_observation_snapshot(observations[0].observation_hash)
    hashes = {
        snapshot.array_content_hash,
        observations[0].array_file_hash,
        *(p.array_file_hash for p in authority.prepared),
    }
    caller = get_ident()
    rendezvous, lock = Barrier(2), Lock()
    reads, checks, trace = Counter(), Counter(), []
    active = peak = 0
    open_file, load = Path.open, np.load

    def counted_open(path, mode="r", *args, **kwargs):
        resolved = Path(path).resolve()
        if "r" in mode and resolved.is_relative_to(store.root.resolve()):
            with lock:
                trace.append((resolved.parent.name, get_ident()))
                if resolved.parent.name in {"lifecycle-arrays", "frozen-observation-arrays"}:
                    reads[resolved.stem] += 1
        return open_file(path, mode, *args, **kwargs)

    @contextmanager
    def checked_archive(archive, identity):
        nonlocal active, peak
        with lock:
            checks[identity] += 1
            ordinal = sum(checks.values())
            active += 1
            peak = max(peak, active)
        try:
            if ordinal <= 2:
                rendezvous.wait(timeout=10)
            yield archive
        finally:
            archive.close()
            with lock:
                active -= 1

    def counted_load(stream, *args, **kwargs):
        archive = load(stream, *args, **kwargs)
        identity = sha256(stream.getvalue()).hexdigest()
        return checked_archive(archive, identity) if identity in hashes else archive

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(np, "load", counted_load)
    admitted = admit_component_inference(root, store=store, expected_hash=authority.content_hash)
    assert admitted.authority == authority
    assert admitted.fit_calls == 0
    assert reads == checks == Counter({identity: 1 for identity in hashes})
    assert peak == 2 and active == 0
    first_worker = next(i for i, (_, thread) in enumerate(trace) if thread != caller)
    assert all(thread == caller for _, thread in trace[:first_worker])
    assert all(
        kind in {"lifecycle-arrays", "frozen-observation-arrays", "frozen-observation-snapshots"}
        and thread != caller
        for kind, thread in trace[first_worker:]
    )
    assert {kind for kind, _ in trace[:first_worker]} >= {
        "lifecycle-training-observations",
        "lifecycle-prepared-refits",
        "frozen-observation-snapshots",
    }


@pytest.mark.parametrize("damaged_source", ("frozen", "labels", "prepared"))
def test_admission_refuses_array_tamper_before_reading_initial_children(
    retained_lifecycle_admission, monkeypatch, damaged_source
):

    from alphalattice.investment.alpha_research.scores.model_renewal import AlphaImportedChild

    store, root, authority, observations = retained_lifecycle_admission
    snapshot = store.load_frozen_observation_snapshot(observations[0].observation_hash)
    hashes = {
        snapshot.array_content_hash,
        observations[0].array_file_hash,
        *(p.array_file_hash for p in authority.prepared),
    }
    prepared = authority.prepared[0]
    child = AlphaImportedChild.create(
        prepared_hash=prepared.content_hash,
        lifecycle_hash=authority.lifecycle.content_hash,
        vintage=prepared.plan.vintage,
        seed=1729,
        estimator_hash="f" * 64,
        payload_hash="f" * 64,
        source_index_hash="f" * 64,
        environment_hash=authority.environment_hash,
    )
    authority = AlphaModelLifecycleAdmission.create(
        **{
            **{
                name: getattr(authority, name)
                for name in type(authority).model_fields
                if name != "content_hash"
            },
            "initial_children": (child,),
        }
    )
    (root / "model-lifecycle-admission.json").write_text(
        authority.model_dump_json(), encoding="utf-8", newline="\n"
    )
    identity = {
        "frozen": snapshot.array_content_hash,
        "labels": observations[0].array_file_hash,
        "prepared": authority.prepared[-1].array_file_hash,
    }[damaged_source]
    category = "frozen-observation-arrays" if damaged_source == "frozen" else "lifecycle-arrays"
    path = store.root / "current" / category / f"{identity}.bin"
    before = path.stat()
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    reads, checks, completed = [], Counter(), Counter()
    rendezvous, lock = Barrier(2), Lock()
    active = 0
    open_file, load = Path.open, np.load

    def counted_open(path, mode="r", *args, **kwargs):
        if "r" in mode:
            reads.append(Path(path).parent.name)
        return open_file(path, mode, *args, **kwargs)

    @contextmanager
    def checked_archive(archive, identity):
        nonlocal active
        with lock:
            checks[identity] += 1
            ordinal = sum(checks.values())
            active += 1
        try:
            if ordinal <= 2:
                rendezvous.wait(timeout=10)
            yield archive
        finally:
            archive.close()
            with lock:
                completed[identity] += 1
                active -= 1

    def counted_load(stream, *args, **kwargs):
        archive = load(stream, *args, **kwargs)
        identity = sha256(stream.getvalue()).hexdigest()
        return checked_archive(archive, identity) if identity in hashes else archive

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(np, "load", counted_load)
    with verified_lifecycle_admissions():
        for _ in range(2):
            reads.clear()
            checks.clear()
            completed.clear()
            with pytest.raises(ValueError, match="frozen_input_content_invalid"):
                admit_component_inference(root, store=store, expected_hash=authority.content_hash)
            assert checks == completed == Counter({value: 1 for value in hashes - {identity}})
            assert active == 0
            assert reads.count("lifecycle-arrays") == 3
            assert reads.count("frozen-observation-arrays") == 1
            assert "imported-models" not in reads


def test_admission_decodes_every_npz_member_without_pickle(
    retained_lifecycle_admission,
):

    store, root, authority, _ = retained_lifecycle_admission
    stream = BytesIO()
    np.savez(stream, features=np.zeros((1, 1)), untrusted=np.array([{}], dtype=object))
    payload = stream.getvalue()
    identity = sha256(payload).hexdigest()
    (store.root / "current/lifecycle-arrays" / f"{identity}.bin").write_bytes(payload)
    prepared = type(authority.prepared[-1]).create(
        **{
            **{
                name: getattr(authority.prepared[-1], name)
                for name in type(authority.prepared[-1]).model_fields
                if name != "content_hash"
            },
            "array_file_hash": identity,
        }
    )
    (store.root / "current/lifecycle-prepared-refits" / f"{prepared.content_hash}.json").write_text(
        prepared.model_dump_json(), encoding="utf-8", newline="\n"
    )
    authority = AlphaModelLifecycleAdmission.create(
        **{
            **{
                name: getattr(authority, name)
                for name in type(authority).model_fields
                if name != "content_hash"
            },
            "prepared": (*authority.prepared[:-1], prepared),
        }
    )
    (root / "model-lifecycle-admission.json").write_text(
        authority.model_dump_json(), encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        admit_component_inference(root, store=store, expected_hash=authority.content_hash)


@pytest.mark.parametrize("temporal", (False, True))
def test_history_and_daily_features_are_identical_and_future_rows_are_inert(temporal):

    source = _source()
    if temporal:
        source = replace(source, reference_eligible=np.ones(source.close.shape, dtype=np.bool_))
    history = prepare_frozen_price_volume_history(
        source,
        ordered_feature_ids=COMPONENT.ordered_feature_ids,
        through=source.formation_sessions[-1],
    )
    for index in (150, 200, 249):
        daily = prepare_frozen_price_volume_features(
            source,
            ordered_feature_ids=COMPONENT.ordered_feature_ids,
            formation_session=source.formation_sessions[index],
        )
        np.testing.assert_array_equal(history[index], daily)
    # A history through an earlier cutoff is the later history's prefix, bit for bit: the
    # training vintages read their rows from one history (AlphaRefitPrices).
    for index in (150, 200):
        earlier = prepare_frozen_price_volume_history(
            source,
            ordered_feature_ids=COMPONENT.ordered_feature_ids,
            through=source.formation_sessions[index],
        )
        np.testing.assert_array_equal(earlier.view("u8"), history[: index + 1].view("u8"))
    if temporal:
        reference = np.ones((len(source.formation_sessions), 9), dtype=np.bool_)
        reference[:-1, -1] = False
        entrant = replace(
            source,
            ordered_listing_ids=(*source.ordered_listing_ids, "entrant"),
            sector_by_listing_id={**source.sector_by_listing_id, "entrant": "sector"},
            reference_eligible=reference,
            **{
                name: np.column_stack((getattr(source, name), getattr(source, name)[:, 0] * 1e8))
                for name in ("open", "high", "low", "close", "volume")
            },
        )
        covered = prepare_frozen_price_volume_history(
            entrant,
            ordered_feature_ids=COMPONENT.ordered_feature_ids,
            through=source.formation_sessions[-1],
        )
        np.testing.assert_array_equal(covered[:-1, :-1].view("u8"), history[:-1].view("u8"))
        assert np.isfinite(covered[-1, -1]).all()  # Own history supplies the initial lookback.


@pytest.mark.parametrize("temporal", (False, True))
def test_maturity_and_budget_refuse_before_fit(tmp_path: Path, monkeypatch, temporal):

    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    source = _source()
    eligible = np.ones(source.close.shape, dtype=np.bool_)
    if temporal:
        eligible[:-1, -1] = False
        source = replace(source, reference_eligible=eligible)
    store = AlphaCurrentArtifactStore(tmp_path)
    obs = store.publish_frozen_observations(source, disposition="SYNTHETIC_INPUT_QA")
    targets = np.log(source.close / source.open)
    labels = publish_training_observations(
        store,
        observation_hash=obs.snapshot_hash,
        target_method_id=COMPONENT.target_recipe,
        target_authority_hash="c" * 64,
        label_available_sessions=tuple(
            day + timedelta(days=1) for day in source.formation_sessions
        ),
        targets=targets,
        eligible=eligible,
    )
    if temporal:
        with pytest.raises(ValueError, match="training_rows_outside_reference"):
            publish_training_observations(
                store,
                observation_hash=obs.snapshot_hash,
                target_method_id=COMPONENT.target_recipe,
                target_authority_hash="c" * 64,
                label_available_sessions=labels.label_available_sessions,
                targets=targets,
                eligible=np.ones(targets.shape, dtype=np.bool_),
            )
    rule = AlphaModelLifecycleRecipe.create(
        **{
            **AlphaModelLifecycleRecipe.from_component(COMPONENT).model_dump(
                exclude={"content_hash"}
            ),
            "month_interval": 1,
            "training_window_sessions": 50,
            "seeds": (1729,),
        }
    )
    plan = resolve_alpha_refit_plan(
        lifecycle=rule,
        vintage="2025-07",
        sessions=source.formation_sessions,
        component_recipe_hash=COMPONENT.recipe_hash,
        source_binding_hash=labels.content_hash,
        ordered_listing_ids=source.ordered_listing_ids,
        ordered_feature_ids=COMPONENT.ordered_feature_ids,
    )
    prepared = prepare_alpha_refit(store, plan=plan, observations=labels, component=COMPONENT)
    if temporal:
        assert prepared.row_count <= len(plan.training_sessions) * 7
    with pytest.raises(ValueError, match="refit_budget_exhausted"):
        fit_alpha_refit_child(
            store, prepared=prepared, component=COMPONENT, seed=1729, budget=AlphaRefitBudget(0)
        )
    late = type(labels).create(
        **{
            **labels.model_dump(exclude={"content_hash"}),
            "label_available_sessions": tuple(date(2026, 1, 1) for _ in source.formation_sessions),
        }
    )
    late_plan = type(plan).create(
        **{
            **plan.model_dump(exclude={"content_hash"}),
            "source_binding_hash": late.content_hash,
        }
    )
    with pytest.raises(ValueError, match="refit_labels_not_mature"):
        prepare_alpha_refit(store, plan=late_plan, observations=late, component=COMPONENT)


@pytest.mark.parametrize("temporal", (False, True))
def test_monthly_yaml_runs_and_replays_through_the_existing_workflow(
    tmp_path, monkeypatch, temporal
):

    from alphalattice.control.research_program.authoring.dispatcher import (
        ResearchExperimentDispatcher,
    )
    from alphalattice.control.research_program.authoring.document import load_authoring_document
    from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
    from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
        AlphaLifecycleExperiment,
    )
    from alphalattice.investment.alpha_research.experiments.verification import (
        AlphaEvidenceVerifier,
    )
    from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
        AlphaRuntimeHeterogeneousPredictionOwner,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AlphaModelLifecycleAdmission,
    )
    from alphalattice.protocols.actor_execution.contracts import ActorKind
    from alphalattice.protocols.research_authoring.contracts import ResolvedResearchAuthority

    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    source = _source()
    store = AlphaCurrentArtifactStore(tmp_path / "source")
    eligible = np.ones(source.close.shape, dtype=np.bool_)
    if temporal:
        eligible[:190, -1] = False
        source = replace(source, reference_eligible=eligible)
    obs = store.publish_frozen_observations(source, disposition="SYNTHETIC_INPUT_QA")
    targets = np.log(source.close / source.open)
    labels = publish_training_observations(
        store,
        observation_hash=obs.snapshot_hash,
        target_method_id=COMPONENT.target_recipe,
        target_authority_hash="c" * 64,
        label_available_sessions=tuple(
            day + timedelta(days=1) for day in source.formation_sessions
        ),
        targets=targets,
        eligible=eligible,
    )
    base = AlphaModelLifecycleRecipe.from_component(COMPONENT)
    admission = AlphaModelLifecycleAdmission.create(
        component=COMPONENT,
        lifecycle=base,
        observations_hash=labels.content_hash,
        prepared=(),
        initial_children=(),
        formation_start=date(2025, 7, 2),
        formation_end=date(2025, 7, 2),
        maximum_fit_attempts=6,
        fit_vintages=("2025-07",),
        environment_hash=AlphaRuntimeHeterogeneousPredictionOwner(
            component_ids=(COMPONENT.component_id,)
        ).environment_hash,
    )
    method = AlphaLifecycleExperiment(store, admission)
    document = load_authoring_document("""
experiment:
  kind: alpha.model-development
  schema_id: research-experiment-envelope
  data_snapshot_handle: component-training.g6_r0_fast_rebound
  universe_handle: us-current-index-research
  sessions: {start: 2025-07-02, end: 2025-07-02,
             as_of: {session: 2025-07-02, phase: OFFICIAL_CLOSE}}
  budget: {maximum_candidates: 20, maximum_numerical_calls: 100}
  determinism: {seed: 1729, thread_limit: 1, network_disabled: true}
  output_workspace: output
alpha:
  methodology_id: MODEL_LIFECYCLE_REPLAY
  component_recipe_id: G6_R0_FAST_REBOUND
  lifecycle: {month_interval: 1, training_window_sessions: 50, seeds: [1729],
              vintage_weights: [1]}
""")

    class Authority:
        def resolve(self, envelope):
            return ResolvedResearchAuthority.create(
                data_snapshot_handle=envelope.data_snapshot_handle,
                universe_handle=envelope.universe_handle,
                training_snapshot_hash=labels.content_hash,
                training_manifest_ref=store.uri(
                    "current/lifecycle-training-observations", labels.content_hash
                ),
                universe_revision_sha256=canonical_hash(source.ordered_listing_ids),
                ordered_listing_ids=source.ordered_listing_ids,
                sessions=(date(2025, 7, 2),),
                source_watermark_hash=obs.snapshot_hash,
            )

    workflow = ResearchProgramWorkflow(
        dispatcher=ResearchExperimentDispatcher(compilers=(method,), authority=Authority()),
        executors=(method,),
        verifiers=(AlphaEvidenceVerifier(),),
        workspace_root=tmp_path,
        source_workspace=tmp_path / "source",
    )
    actor = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    sealed, _ = workflow.preflight(document, **actor)
    denied = AlphaLifecycleExperiment(
        store,
        type(admission).create(
            **{
                **admission.model_dump(exclude={"content_hash"}),
                "fit_vintages": (),
            }
        ),
    )
    denied_dispatcher = ResearchExperimentDispatcher(compilers=(denied,), authority=Authority())
    with pytest.raises(ValueError, match="refit_period_not_admitted"):
        denied_dispatcher.compile(document)
    # Count the first array read as well as the subsequent proof-backed readbacks.
    opened: Counter[Path] = Counter()
    written: Counter[Path] = Counter()
    open_file = Path.open

    def counted_open(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "rb":
            opened[file] += 1
        if any(flag in mode for flag in "wax+") and any(
            file.is_relative_to(tmp_path / root) for root in ("source", "output")
        ):
            written[file] += 1
        return open_file(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    evidence, _ = workflow.run_sealed(document, **actor)
    assert evidence.program_hash == sealed.program_hash
    # One fit plus one score; this child's fit stores no training-error prediction.
    assert evidence.numerical_call_count == 2
    assert written  # The counter observed the producer before checking replay.
    written.clear()
    replay, _ = workflow.replay(document, **actor)
    assert not written, (
        f"COUNT STOP op=replay_writable_open observed={dict(written)} budget=0 "
        "fixture=monthly-yaml wayon=reduce-work"
    )
    assert replay.numerical_call_count == 0
    assert replay.artifact_uris == evidence.artifact_uris
    bad = dict(document)
    bad["alpha"] = {**document["alpha"], "lifecycle": {"callback": "pretend"}}
    with pytest.raises(ValueError, match="parameter_not_installed"):
        workflow.preflight(bad, **actor)

    # Recursive readback must reach the numerical score bytes, not just receipts.
    from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
        CATEGORY,
        AlphaLifecycleResearchReceipt,
    )

    output_store = AlphaCurrentArtifactStore(tmp_path / "output/alpha-lifecycle")
    identity = output_store._hash_from_uri(evidence.artifact_uris[0], "current/" + CATEGORY)
    receipt = output_store._load(CATEGORY, identity, "content_hash", AlphaLifecycleResearchReceipt)
    path = output_store._path("current/lifecycle-arrays", receipt.projection_files[0], "bin")
    assert opened[path] >= 1  # the run and its replay read the array in full
    original = path.read_bytes()
    # Complete readbacks retain the OS proof on Windows; other platforms read again.
    from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
        verify_lifecycle_research,
    )

    seeded = opened[path]
    for _ in range(2):
        before_read = opened[path]
        assert (
            verify_lifecycle_research(
                program=sealed, evidence=evidence, output_workspace=tmp_path / "output"
            )
            == "CURRENT"
        )
        assert opened[path] == (seeded if os.name == "nt" else before_read + 1)

    before = path.stat()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="frozen_input_content_invalid"):
        workflow.replay(document, **actor)
    path.write_bytes(original)
    written.clear()
    restored, _ = workflow.replay(document, **actor)
    assert not written, (
        f"COUNT STOP op=restored_replay_writable_open observed={dict(written)} budget=0 "
        "fixture=monthly-yaml wayon=reduce-work"
    )
    assert restored.numerical_call_count == 0
    assert restored.artifact_uris == evidence.artifact_uris


def test_legacy_panel_and_package_component_identities_do_not_rotate():
    from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
        ComponentPlanEntry,
    )
    from alphalattice.protocols.research_authoring.contracts import ResolvedResearchAuthority

    payload = {
        "data_snapshot_handle": "current",
        "universe_handle": "us-current-index-research",
        "panel_snapshot_hash": "1" * 64,
        "panel_manifest_ref": "playpen://panel/fixture",
        "universe_revision_sha256": "2" * 64,
        "ordered_listing_ids": ["one", "two"],
        "sessions": ["2026-07-01"],
        "source_watermark_hash": "3" * 64,
    }
    legacy = ResolvedResearchAuthority(**payload, authority_hash=canonical_hash(payload))
    assert legacy.model_dump(mode="json", exclude={"authority_hash"}) == payload
    with pytest.raises(ValueError, match="source_authority_shape_invalid"):
        ResolvedResearchAuthority.create(
            **payload, training_snapshot_hash="4" * 64, training_manifest_ref="training"
        )
    component = {
        "kind": "ComponentPlanEntry",
        "component_id": "component",
        "allocation_basis_points": 10000,
        "family_id": "family",
        "recipe_hash": "1" * 64,
        "target_recipe": "target",
        "objective": "objective",
    }
    assert ComponentPlanEntry(**component).model_dump(mode="json") == component


def test_formula_snapshot_roundtrip_is_bound_without_changing_legacy_shape(tmp_path):

    source = _source()
    store = AlphaCurrentArtifactStore(tmp_path)
    old = store.publish_frozen_observations(source, disposition="SYNTHETIC_INPUT_QA")
    assert "ordered_formula_ids" not in old.model_dump(mode="json")
    assert "reference_eligibility_recorded" not in old.model_dump(mode="json")
    assert "nominal_population_recorded" not in old.model_dump(mode="json")
    assert store.load_frozen_observations(old.snapshot_hash).reference_eligible is None
    reference = np.ones(source.close.shape, dtype=np.bool_)
    reference[:-1, -1] = False
    enriched = replace(
        source,
        formula_values={"mom_252_21": source.close.copy()},
        reference_eligible=reference,
        nominal_member_count=np.full(len(source.formation_sessions), 8, dtype=np.int64),
    )
    new = store.publish_frozen_observations(enriched, disposition="SYNTHETIC_INPUT_QA")
    assert old.snapshot_hash != new.snapshot_hash
    reopened = store.load_frozen_observations(new.snapshot_hash)
    np.testing.assert_array_equal(reopened.formula_values["mom_252_21"], source.close)
    assert not reopened.formula_values["mom_252_21"].flags.writeable
    np.testing.assert_array_equal(reopened.reference_eligible, reference)
    np.testing.assert_array_equal(reopened.nominal_member_count, enriched.nominal_member_count)
    assert not reopened.reference_eligible.flags.writeable
    assert not reopened.nominal_member_count.flags.writeable
    all_members = replace(enriched, reference_eligible=np.ones(reference.shape, dtype=np.bool_))
    assert (
        store.publish_frozen_observations(
            all_members, disposition="SYNTHETIC_INPUT_QA"
        ).snapshot_hash
        != new.snapshot_hash
    )
    with pytest.raises(ValueError, match="prefix_changed"):
        all_members.require_unchanged_prefix(reopened)
    with pytest.raises(ValueError, match="prefix_changed"):
        replace(reopened, formula_values={"mom_252_21": source.close + 1}).require_unchanged_prefix(
            reopened
        )
    path = store._path("current/frozen-observation-arrays", new.array_content_hash, "bin")
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="frozen_input_content_invalid"):
        store.load_frozen_observations(new.snapshot_hash)


@pytest.mark.parametrize(
    "method,horizon",
    (("EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET", 1), ("T1_H3_PURE_TOTAL_RETURN_Z", 3)),
)
def test_component_target_maturity_does_not_consume_unobserved_exits(method, horizon):
    from alphalattice.investment.alpha_research.targets.component_training import (
        compile_frozen_component_training_targets,
    )

    source = _source()
    days = source.formation_sessions[:10]
    raw = np.random.default_rng(31415).normal(0, 0.01, (10, 8))
    ends = tuple(days[i + 2] if i + 2 < len(days) else None for i in range(len(days)))
    values, available = compile_frozen_component_training_targets(
        source=source,
        formation_sessions=days,
        holding_end_sessions=ends,
        raw_log_returns=raw,
        simple_returns=np.expm1(raw),
        target_method_id=method,
    )
    assert available[0] == days[horizon + 1]
    assert np.isnan(values[-horizon - 1 :]).all()
    changed = raw.copy()
    changed[-1] = 100
    later, _ = compile_frozen_component_training_targets(
        source=source,
        formation_sessions=days,
        holding_end_sessions=ends,
        raw_log_returns=changed,
        simple_returns=np.expm1(changed),
        target_method_id=method,
    )
    np.testing.assert_array_equal(values[:5], later[:5])
    with pytest.raises(ValueError, match="target_axis_invalid"):
        compile_frozen_component_training_targets(
            source=source,
            formation_sessions=days,
            holding_end_sessions=(days[1], *ends[1:]),
            raw_log_returns=raw,
            simple_returns=np.expm1(raw),
            target_method_id=method,
        )


def test_source_successor_keeps_old_period_and_cannot_reset_fit_permission(tmp_path):
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AlphaModelLifecycleAdmission,
        renew_lifecycle_admission,
        verify_lifecycle_successor,
    )

    store = AlphaCurrentArtifactStore(tmp_path)
    source = _source(380)
    snapshot = store.publish_frozen_observations(source, disposition="SYNTHETIC_INPUT_QA")
    targets = np.log(source.close / source.open)
    args = dict(
        observation_hash=snapshot.snapshot_hash,
        target_method_id=COMPONENT.target_recipe,
        target_authority_hash="a" * 64,
        label_available_sessions=tuple(
            day + timedelta(days=1) for day in source.formation_sessions
        ),
        targets=targets,
        eligible=np.ones(targets.shape, dtype=np.bool_),
    )
    old = publish_training_observations(store, **args)
    rule = AlphaModelLifecycleRecipe.create(
        **{
            **AlphaModelLifecycleRecipe.from_component(COMPONENT).model_dump(
                exclude={"content_hash"}
            ),
            "training_window_sessions": 50,
            "vintage_weights": (1,),
            "seeds": (1729,),
        }
    )
    plan = resolve_alpha_refit_plan(
        lifecycle=rule,
        vintage="2025-07",
        sessions=source.formation_sessions,
        component_recipe_hash=COMPONENT.recipe_hash,
        source_binding_hash=old.content_hash,
        ordered_listing_ids=source.ordered_listing_ids,
        ordered_feature_ids=COMPONENT.ordered_feature_ids,
    )
    prepared = prepare_alpha_refit(store, plan=plan, observations=old, component=COMPONENT)
    base = AlphaModelLifecycleAdmission.create(
        component=COMPONENT,
        lifecycle=rule,
        observations_hash=old.content_hash,
        prepared=(prepared,),
        initial_children=(),
        formation_start=date(2025, 7, 1),
        formation_end=date(2025, 7, 1),
        maximum_fit_attempts=0,
        environment_hash="b" * 64,
        fit_vintages=("2025-07", "2025-10"),
        training_factor_ids=("fixture",),
        renewal_through=date(2025, 12, 1),
    )
    newer = publish_training_observations(store, **args, training_support_hash="c" * 64)
    updated = renew_lifecycle_admission(
        store, previous=base, observations=newer, through=date(2025, 7, 2)
    )
    assert updated.observations_hash != base.observations_hash
    assert updated.prepared == base.prepared
    assert updated.budget_binding == base.budget_binding
    verify_lifecycle_successor(store, ancestor=base, successor=updated)
    next_quarter = renew_lifecycle_admission(
        store, previous=updated, observations=newer, through=date(2025, 10, 1)
    )
    assert next_quarter.prepared[0] == base.prepared[0]
    assert tuple(p.plan.vintage for p in next_quarter.prepared) == ("2025-07", "2025-10")
    assert next_quarter.budget_binding == base.budget_binding
    assert next_quarter.maximum_fit_attempts == 0
    verify_lifecycle_successor(store, ancestor=base, successor=next_quarter)
    bad = type(updated).create(
        **{**updated.model_dump(exclude={"content_hash"}), "maximum_fit_attempts": 1}
    )
    with pytest.raises(ValueError, match="permission_changed"):
        verify_lifecycle_successor(store, ancestor=base, successor=bad)
    changed = targets.copy()
    changed[180] += 1
    correction = publish_training_observations(
        store, **{**args, "targets": changed}, training_support_hash="c" * 64
    )
    with pytest.raises(ValueError, match="correction_not_admitted"):
        renew_lifecycle_admission(
            store, previous=base, observations=correction, through=date(2025, 7, 2)
        )
    revised = renew_lifecycle_admission(
        store,
        previous=base,
        observations=correction,
        through=date(2025, 7, 2),
        source_policy="REVISED_INPUTS_NEW_VINTAGES_ONLY",
    )
    assert revised.prepared == base.prepared
    assert revised.maximum_fit_attempts == 0
    assert revised.source_transition_hash == canonical_hash(
        [
            base.observations_hash,
            correction.content_hash,
            "REVISED_INPUTS_NEW_VINTAGES_ONLY",
        ]
    )
    assert revised.budget_binding == base.budget_binding
    verify_lifecycle_successor(store, ancestor=base, successor=revised)


@pytest.mark.parametrize("temporal", (False, True))
def test_score_dispatch_uses_captured_formation_membership(tmp_path, monkeypatch, temporal):

    from alphalattice.control.product_host.composition.strategy_scoring import (
        STAGES,
        StrategyScoringApplication,
    )

    source = _source()
    expected = np.ones(source.close.shape[1], dtype=np.bool_)
    if temporal:
        mask = np.ones(source.close.shape, dtype=np.bool_)
        mask[:-1, -1] = False
        source = replace(source, reference_eligible=mask)
        expected[-1] = False
    store = AlphaCurrentArtifactStore(tmp_path)
    observed = store.publish_frozen_observations(source, disposition="SYNTHETIC_INPUT_QA")
    formation = source.formation_sessions[-2]  # Not the later/current mask.
    captured = []

    class DispatchObserved(Exception):
        pass

    def score(**kwargs):
        captured.append(kwargs)
        raise DispatchObserved  # Stop before prediction/publication; this tests the boundary.

    owner = object.__new__(StrategyScoringApplication)
    monkeypatch.setattr(owner, "_require_plan", lambda _: None)
    monkeypatch.setattr(owner, "_published", lambda _: None)
    monkeypatch.setattr(owner, "_captured_authority", lambda *_: SimpleNamespace(score=score))
    owner.store = SimpleNamespace(
        load_frozen_feature_preparation=lambda _: (
            SimpleNamespace(
                inference_authority_hash="a" * 64,
                observation_snapshot_hash=observed.snapshot_hash,
                ordered_listing_ids=source.ordered_listing_ids,
            ),
            (),
        ),
        load_frozen_observations=store.load_frozen_observations,
    )
    with pytest.raises(DispatchObserved):
        owner.execute_step(
            SimpleNamespace(formation_session=formation, binding=None),
            STAGES[2],
            lambda _: "b" * 64,
        )
    assert len(captured) == 1
    assert captured[0]["formation"] == formation
    np.testing.assert_array_equal(captured[0]["eligible"], expected)


def test_context_history_uses_each_dates_reference_not_the_latest_roster(monkeypatch):

    from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
        assemble_panel_context_arrays,
    )
    from alphalattice.investment.sector_research.inputs import surface
    from alphalattice.investment.sector_research.inputs.surface import compile_sector_context_arrays
    from alphalattice.kernel.shared_kernel.spans import collect, readout

    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(300))
    ends = tuple(day + timedelta(days=1) for day in days)
    values = np.random.default_rng(87).normal(0, 0.01, (len(days), 12))

    def context(raw, mask):
        ids = tuple(f"L{i:03d}" for i in range(raw.shape[1]))
        sectors = MappingProxyType({name: f"S{i % 2}" for i, name in enumerate(ids)})
        with collect() as ledger:
            state, identity = compile_sector_context_arrays(
                sessions=days,
                holding_end_sessions=ends,
                listing_ids=ids,
                raw_log_returns=raw,
                target_evidence_hash="a" * 64,
                sector_by_listing_id=sectors,
                sector_revision="b" * 64,
                reference_eligible=mask,
            )
        work = {row["detail"]: row["count"] for row in readout(ledger)["spans"]}
        for operation, budget in (
            ("sector_axes", len(days) + len(ids)),
            ("sector_clocks", 2 * len(days)),
            ("sector_causal", len(days)),
            ("sector_rows", len(days) * len(set(sectors.values()))),
        ):
            observed = work[operation]
            assert observed <= budget, (
                f"COUNT STOP op={operation} observed={observed} budget={budget} "
                "way_on=reduce_context_before_python"
            )
        sector, market = assemble_panel_context_arrays(
            formation_sessions=days,
            holding_end_sessions=ends,
            ordered_listing_ids=ids,
            ordered_sector_ids=("S0", "S1"),
            sector_by_listing_id=sectors,
            raw_log_execution_returns=raw,
            raw_simple_execution_returns=np.expm1(raw),
            sector_state_values=state,
            market_interaction_state_values=np.ones((len(days), 2)),
            observation_returns=raw,
            observation_volume_state=raw + 1,
            observation_dollar_volume=np.abs(raw) * 1e6,
            observation_high_distance=-np.abs(raw),
            observation_low_distance=np.abs(raw),
            reference_eligible=mask,
        )
        return state, sector, market, identity

    before = context(values, np.ones(values.shape, dtype=np.bool_))
    extended = np.column_stack((values, np.full(len(days), 0.5)))
    mask = np.ones(extended.shape, dtype=np.bool_)
    mask[:-1, -1] = False
    after = context(extended, mask)
    unique = surface.pc.unique

    def repeated_listing_objects(column):
        distinct = unique(column)
        if surface.pa.types.is_string(distinct.type):
            return surface.pa.concat_arrays((distinct, distinct))
        return distinct

    with monkeypatch.context() as repeated:
        repeated.setattr(surface.pc, "unique", repeated_listing_objects)
        with pytest.raises(AssertionError, match="COUNT STOP op=sector_axes"):
            context(values, np.ones(values.shape, dtype=np.bool_))
    make_table = surface.pa.table

    def divergent_clock(columns, *args, **kwargs):
        clocks = columns["holding_end_open_at"].to_pylist()
        clocks[1] += timedelta(seconds=1)
        return make_table({**columns, "holding_end_open_at": clocks}, *args, **kwargs)

    with monkeypatch.context() as divergence:
        divergence.setattr(surface.pa, "table", divergent_clock)
        with pytest.raises(surface.SectorContextBoundaryError, match="clock_mismatch"):
            context(values, np.ones(values.shape, dtype=np.bool_))
    for baseline, result in zip(before[:3], after[:3], strict=True):
        np.testing.assert_array_equal(baseline[:-1].view("u8"), result[:-1].view("u8"))
    assert not np.array_equal(before[2][-1], after[2][-1])
    # The entrant has a full own-price volatility history; it need not wait
    # 63 sessions after entry to participate in today's implied-correlation input.
    assert np.isfinite(after[2][-1, 10])
    assert before[3] != after[3]  # Reference membership participates in provenance.

    from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
        materialize_panel_feature_projection,
        preflight_panel_feature_plan,
    )
    from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
        PanelFeatureSourceArrays,
    )

    def project(raw, reference, contexts):

        from alphalattice.investment.alpha_research.experiments.panel_alpha_fold_execution import (
            PanelFeatureSourcePayload,
        )

        ids = tuple(f"L{i:03d}" for i in range(raw.shape[1]))
        matrices = [raw[:, :, None], raw, np.expm1(raw)]
        for matrix in matrices:
            matrix.setflags(write=False)
        reference.setflags(write=False)
        source = PanelFeatureSourceArrays(
            formation_sessions=days,
            holding_end_sessions=ends,
            ordered_listing_ids=ids,
            ordered_factor_ids=("mom_252_21",),
            absolute_state_factor_ids=("mom_252_21",),
            ordered_sector_ids=("S0", "S1"),
            sector_by_listing_id=MappingProxyType(
                {name: f"S{i % 2}" for i, name in enumerate(ids)}
            ),
            raw_formula_values=matrices[0],
            total_return_target_z=raw,
            raw_log_execution_returns=raw,
            raw_simple_execution_returns=matrices[2],
            sector_context_values=contexts[1],
            market_context_values=contexts[2],
            source_identity_hashes=MappingProxyType({"fixture": "c" * 64}),
            reference_eligible=reference,
        )
        plan = preflight_panel_feature_plan(
            source=source, selected_method_ids=("RELATIVE_CONTROL",), maximum_aggregation_span=1
        )
        restored = pickle.loads(
            pickle.dumps(PanelFeatureSourcePayload.from_source(source))
        ).restore()
        restored_plan = preflight_panel_feature_plan(
            source=restored,
            selected_method_ids=("RELATIVE_CONTROL",),
            maximum_aggregation_span=1,
        )
        assert restored_plan.preflight == plan.preflight
        np.testing.assert_array_equal(restored.reference_eligible, source.reference_eligible)
        result = materialize_panel_feature_projection(
            plan=plan,
            method_id="RELATIVE_CONTROL",
            program_hash="d" * 64,
            fold_index=0,
            boundary_id="OUTER",
            training_sessions=days[100:200],
            transform_sessions=days[250:],
        )
        return plan, result

    original_plan, original_projection = project(
        values, np.ones(values.shape, dtype=np.bool_), before
    )
    extended_plan, extended_projection = project(extended, mask, after)
    assert not extended_plan.common_row_mask[:-1, -1].any()
    assert extended_plan.common_row_mask[-1, -1]
    np.testing.assert_array_equal(
        original_plan.common_row_mask, extended_plan.common_row_mask[:, :12]
    )
    np.testing.assert_array_equal(
        original_projection.training_features.view("u8"),
        extended_projection.training_features.view("u8"),
    )
    assert "L012" not in extended_projection.row_listing_ids[:-13]  # No historical entrant samples.


def test_legacy_score_identity_readback_does_not_reopen_model_authority(monkeypatch):

    from alphalattice.control.product_host.composition.strategy_scoring import (
        StrategyScoringApplication,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy score readback must not require model authority files")

    monkeypatch.setattr(StrategyScoringApplication, "_captured_authority", forbidden)
    owner = object.__new__(StrategyScoringApplication)
    score = SimpleNamespace(model_set_publication_hash=None, inference_authority_hash="a" * 64)
    owner._verify_model_publication(score, SimpleNamespace(authority_hash="a" * 64))
    with pytest.raises(ValueError, match="publication_evidence_mismatch"):
        owner._verify_model_publication(score, SimpleNamespace(authority_hash="b" * 64))


def test_the_full_lifecycle_is_the_components_own_and_the_light_default_trains_one_seed() -> None:
    """FULL keeps the lifecycle hash; LIGHT fits one seed per vintage on the same calendar."""
    assert DEFAULT_MODEL_LIFECYCLE == "LIGHT"
    for name in ("G2_R0_TREND", "G6_R0_FAST_REBOUND"):
        component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(name)
        full = AlphaModelLifecycleRecipe.named(component, "FULL")
        light = AlphaModelLifecycleRecipe.named(component, "LIGHT")
        assert full == AlphaModelLifecycleRecipe.from_component(component)
        assert len(full.seeds) == 3 and light.seeds == tuple(component.seeds[:1])
        kept = {"content_hash", "seeds"}
        assert light.model_dump(exclude=kept) == full.model_dump(exclude=kept)
        assert model_lifecycle_of(component, light.content_hash) == "LIGHT"
        assert model_lifecycle_of(component, full.content_hash) == "FULL"
        assert model_lifecycle_of(component, "0" * 64) is None
