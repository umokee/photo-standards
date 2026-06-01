import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import type { GroupDetail } from "@/types/contracts";
import { useEffect, useState } from "react";
import { buildUpdateGroupPayload, useUpdateGroup } from "../api/update-group";

export const UpdateGroup = ({ group }: { group: GroupDetail }) => (
  <Modal>
    <Modal.Trigger>
      <Button variant="ghost">Изменить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <UpdateGroupModal group={group} />
    </Modal.Content>
  </Modal>
);

const UpdateGroupModal = ({ group }: { group: GroupDetail }) => {
  const [name, setName] = useState(group.name);
  const [description, setDescription] = useState(group.description ?? "");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const close = useModalClose();
  const mutation = useUpdateGroup();

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  const handleSubmit = () => {
    const result = buildUpdateGroupPayload(
      { name, description },
      { name: group.name, description: group.description ?? "" }
    );

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    if (!result.data) return;

    mutation.mutate({ id: group.id, data: result.data });
  };

  return (
    <>
      <Modal.Header>{`Изменить группу ${group.name}`}</Modal.Header>
      <Modal.Body>
        <Input
          label="Название"
          placeholder="Пример"
          value={name}
          onChange={setName}
          error={formErrors.name ?? getFieldError(mutation.error, "name")}
        />

        <Input
          label="Описание"
          placeholder="Пример"
          value={description}
          onChange={setDescription}
          error={formErrors.description ?? getFieldError(mutation.error, "description")}
        />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>

        <Button disabled={mutation.isPending} onClick={handleSubmit}>
          Cохранить
        </Button>
      </Modal.Footer>
    </>
  );
};
