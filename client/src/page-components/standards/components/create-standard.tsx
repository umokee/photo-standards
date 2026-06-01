import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import { useAngleOptions } from "@/constants";
import { getFieldError } from "@/lib/errors";
import type { Angle } from "@/types/contracts";
import { useEffect, useState } from "react";
import { buildCreateStandardPayload, useCreateStandard } from "../api/create-standard";

export const CreateStandard = ({ groupId }: { groupId: string }) => (
  <Modal>
    <Modal.Trigger>
      <Button>Новый эталон</Button>
    </Modal.Trigger>
    <Modal.Content>
      <CreateStandardModal groupId={groupId} />
    </Modal.Content>
  </Modal>
);

const CreateStandardModal = ({ groupId }: { groupId: string }) => {
  const close = useModalClose();
  const angleOptions = useAngleOptions();
  const [name, setName] = useState("");
  const [angle, setAngle] = useState<Angle | null>(null);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const mutation = useCreateStandard();

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  const handleSubmit = () => {
    const result = buildCreateStandardPayload({ groupId, name, angle });

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    mutation.mutate(result.data);
  };

  return (
    <>
      <Modal.Header>Добавить эталон</Modal.Header>
      <Modal.Body>
        <Input
          label="Название"
          placeholder="Пример"
          value={name}
          onChange={setName}
          error={formErrors.name ?? getFieldError(mutation.error, "name")}
        />
        <Select
          label="Ракурс"
          options={angleOptions}
          value={angle ?? ""}
          placeholder="Выберите ракурс"
          onChange={(val) => setAngle(val ? (val as Angle) : null)}
          error={formErrors.angle ?? getFieldError(mutation.error, "angle")}
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
