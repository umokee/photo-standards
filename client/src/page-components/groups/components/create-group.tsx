import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import { useEffect, useState } from "react";
import { buildCreateGroupPayload, useCreateGroup } from "../api/create-group";

export const CreateGroup = () => (
  <Modal>
    <Modal.Trigger>
      <Button variant="ghost" full size="sm">
        Новая группа
      </Button>
    </Modal.Trigger>
    <Modal.Content>
      <CreateGroupModal />
    </Modal.Content>
  </Modal>
);

const CreateGroupModal = () => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const close = useModalClose();
  const mutation = useCreateGroup();

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  const handleSubmit = () => {
    const result = buildCreateGroupPayload({ name, description });

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    mutation.mutate(result.data);
  };

  return (
    <>
      <Modal.Header>Создать группу</Modal.Header>
      <Modal.Body>
        <Input
          label="Название"
          value={name}
          onChange={setName}
          error={formErrors.name ?? getFieldError(mutation.error, "name")}
        />

        <Input
          label="Описание"
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
          Создать
        </Button>
      </Modal.Footer>
    </>
  );
};
