use crate::operator::model::*;
use crate::session::repository::SessionRepository;
use crate::shared::utils;

use dioxus::signals::{WritableExt, WriteSignal};
use uuid::Uuid;

const OPERATOR_KEY: &str = "operator";

#[derive(Clone, Debug)]
pub(crate) struct OperatorRepository {
    operators: Vec<(Uuid, Operator)>,
}

impl OperatorRepository {
    fn save(&self) -> anyhow::Result<()> {
        utils::set_item(OPERATOR_KEY, &self.operators)
    }

    pub(crate) fn from_local_storage_or_default() -> anyhow::Result<OperatorRepository> {
        let operators = utils::get_item_or_default(OPERATOR_KEY, Vec::<(Uuid, Operator)>::new)?;

        Ok(OperatorRepository { operators })
    }

    pub(crate) fn push_operator(&mut self, operator: Operator) -> anyhow::Result<Uuid> {
        let operator_uuid = Uuid::new_v4();

        self.operators.push((operator_uuid, operator));
        self.save()?;

        Ok(operator_uuid)
    }

    pub(crate) fn deactivate_operator(
        &mut self,
        operator_uuid: Uuid,
        mut sessions: WriteSignal<SessionRepository>,
    ) -> anyhow::Result<()> {
        if let Some((_, operator)) = self.operators.iter_mut().find(|x| x.0 == operator_uuid) {
            operator.deactivate();
            sessions
                .write()
                .deactivate_operator_helper(operator_uuid, self.operators())?;
            self.save()?;
        } else {
            anyhow::bail!("Uuid `{operator_uuid}` is not found in OperatorRepository.operators");
        }

        Ok(())
    }

    pub(crate) fn rename(&mut self, operator_uuid: Uuid, new_name: String) -> anyhow::Result<()> {
        if let Some((_, operator)) = self.operators.iter_mut().find(|x| x.0 == operator_uuid) {
            operator.rename(new_name);
            self.save()?;
        } else {
            anyhow::bail!("Uuid `{operator_uuid}` is not found in OperatorRepository.operators");
        }

        Ok(())
    }

    pub(crate) fn operators(&self) -> &Vec<(Uuid, Operator)> {
        &self.operators
    }

    pub(crate) fn iterator(&self) -> impl Iterator<Item = (Uuid, &Operator)> {
        self.operators
            .iter()
            .filter(|x| x.1.activity())
            .map(|(uuid, operator)| (*uuid, operator))
    }

    pub(crate) fn get(&self, uuid: Uuid) -> anyhow::Result<&Operator> {
        if let Some(operator) = self.operators.iter().find(|x| x.0 == uuid) {
            Ok(&operator.1)
        } else {
            anyhow::bail!("Failed to get uuid `{uuid}` in OperatorRepository.operators");
        }
    }
}
