import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import type { GroupDetail } from "@/types/contracts";
import { useEffect, useState } from "react";
import { buildUpdateGroupPayload, useUpdateGroup } from "../api/update-group";

type GroupActionTarget = Pick<GroupDetail, "id" | "name" | "description">;

type UpdateGroupProps = {
  group: GroupActionTarget;
  triggerClassName?: string;
};

export const UpdateGroup = ({ group, triggerClassName }: UpdateGroupProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} variant="ghost" size="sm" aria-label={`Изменить изделие ${group.name}`}>Изменить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <UpdateGroupModal group={group} />
    </Modal.Content>
  </Modal>
);

const UpdateGroupModal = ({ group }: { group: GroupActionTarget }) => {
  const [name, setName] = useState(group.name);
  const [description, setDescription] = useState(group.description ?? "");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const close = useModalClose();
  const mutation = useUpdateGroup();

  useEffect(() => {
    setName(group.name);
    setDescription(group.description ?? "");
    setFormErrors({});
  }, [group.description, group.name]);

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [close, mutation.isSuccess]);

  const handleSubmit = () => {
    const result = buildUpdateGroupPayload(
      { name, description },
      { name: group.name, description: group.description ?? "" }
    );

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    if (!result.data) {
      close();
      return;
    }

    setFormErrors({});
    mutation.mutate({ id: group.id, data: result.data });
  };

  return (
    <>
      <Modal.Header>Изменить изделие</Modal.Header>
      <Modal.Body>
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
          placeholder="Что проверяем и какие ракурсы нужны"
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

        {formErrors.form ? <div>{formErrors.form}</div> : null}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>Отмена</Button>
        <Button disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Сохранение..." : "Сохранить"}
        </Button>
      </Modal.Footer>
    </>
  );
};
