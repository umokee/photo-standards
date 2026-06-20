import { paths } from "@/app/paths";
import p from "./group-modal.module.scss";
import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import { CheckCircle2, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { buildCreateGroupPayload, useCreateGroup } from "../api/create-group";

export const CreateGroup = () => (
  <Modal>
    <Modal.Trigger>
      <Button icon={Plus} size="sm">
        Новое изделие
      </Button>
    </Modal.Trigger>
    <Modal.Content wide>
      <CreateGroupModal />
    </Modal.Content>
  </Modal>
);

const CreateGroupModal = () => {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const close = useModalClose();
  const mutation = useCreateGroup();

  useEffect(() => {
    if (mutation.isSuccess) {
      const groupId = mutation.data?.id;
      close();
      if (groupId) navigate(paths.groupDetail(groupId));
    }
  }, [close, mutation.data?.id, mutation.isSuccess, navigate]);

  const handleSubmit = () => {
    const result = buildCreateGroupPayload({ name, description });

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    setFormErrors({});
    mutation.mutate(result.data);
  };

  return (
    <>
      <Modal.Header>Новое изделие</Modal.Header>
      <Modal.Body>
        <div className={p.modalFormGrid}>
          <Input
            label="Название"
            placeholder="Например: Корпус редуктора"
            value={name}
            onChange={(value) => {
              setName(value);
              setFormErrors((current) => {
                const next = { ...current };
                delete next.name;
                delete next.form;
                return next;
              });
            }}
            error={formErrors.name ?? getFieldError(mutation.error, "name")}
          />

          <Input
            label="Описание"
            placeholder="Что проверяем, какие ракурсы нужны, особенности изделия"
            value={description}
            onChange={(value) => {
              setDescription(value);
              setFormErrors((current) => {
                const next = { ...current };
                delete next.description;
                delete next.form;
                return next;
              });
            }}
            error={formErrors.description ?? getFieldError(mutation.error, "description")}
          />
        </div>

        {formErrors.form ? <div className={p.modalError}>{formErrors.form}</div> : null}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>

        <Button icon={CheckCircle2} disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Создание..." : "Создать"}
        </Button>
      </Modal.Footer>
    </>
  );
};

