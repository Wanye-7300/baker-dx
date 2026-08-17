use crate::operator::repository::*;

use dioxus::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct OperatorViewModel {
    pub(crate) operator_repository: Signal<OperatorRepository>,
}

impl OperatorViewModel {
    pub(crate) fn use_operator_view_model_provider() -> anyhow::Result<()> {
        let operator_repository = OperatorRepository::from_local_storage_or_default()?;
        let operator_repository = use_signal(|| operator_repository);

        use_context_provider(|| OperatorViewModel { operator_repository });

        Ok(())
    }
}
