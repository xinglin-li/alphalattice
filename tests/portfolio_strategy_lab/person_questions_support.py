from alphalattice.interface.local_application import questions


def question_row(at, state, *, goal="goal-a", session="session-a", catalog=None):
    definition = questions.person_question("network.workspace_access")
    refs = dict(operation_ref=at, agent_session=session)
    if goal is not None:
        refs["goal_id"] = goal
    question = dict(
        code=definition["code"],
        catalog_hash=catalog or definition["catalog_hash"],
        state=state,
        references=refs,
    )
    return dict(
        schema_kind="ProductOperationObserved",
        source_kind="PRODUCT_OPERATION",
        authority="OPERATIONAL_ASSERTION",
        correlation_ids=[at],
        payload=dict(phase="RETURNED", operation_ref=at, subject=dict(person_questions=[question])),
    )
