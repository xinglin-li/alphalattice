"""Task dispatch identifiers shared with the Host's lightweight route surface."""

CAPTURE_TASK_KIND = "research_input_capture"
DECISION_ADVANCEMENT_SCHEMA = "portfolio-conditional-decision-advancement"
EVIDENCE_INSTALL_TASK_KIND = "evidence_install"
EVIDENCE_REVIEW_RUNTIME_FILE = "evidence-review-checkpoints.sqlite"
FEATURE_BUILD_TASK_KIND = "research_feature_materialization"
MODEL_TRAINING_INPUT_TASK_KIND = "model_training_input_preparation"
PORTFOLIO_UPDATE_TASK_KIND = "portfolio_decision_update"
RESEARCH_EXPERIMENT_TASK_KIND = "research_experiment"
RESEARCH_STRATEGY_TASK_KIND = "research_strategy_preparation"
STRATEGY_CALIBRATION_TASK_KIND = "portfolio_calibration_preparation"
STRATEGY_SCORE_TASK_KIND = "strategy_score_preparation"
STUDY_VERIFICATION_TASK_KIND = "study_verification_sweep"
WORKSPACE_DATA_UPDATE_TASK_KIND = "workspace_data_update"
WORKSPACE_PREPARATION_TASK_KIND = "workspace_preparation"
"""The Host-owned commands' Task kinds as the recovery catalog keys them without importing a
command; each equals its command's own `command_kind`, which the catalog checks as it builds it."""


VERIFICATION_LEDGER_DIRECTORY = "verification-ledger"
