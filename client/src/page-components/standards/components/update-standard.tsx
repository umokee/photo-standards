import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import { useAngleOptions } from "@/constants";
import { getFieldError } from "@/lib/errors";
import type { Angle, GroupStandard } from "@/types/contracts";
import { useEffect, useState } from "react";
import { buildUpdateStandardPayload, useUpdateStandard } from "../api/update-standard";

type EditableStandard = Pick<GroupStandard, "id" | "group_id" | "name" | "angle">;

type UpdateStandardProps = {
  standard: EditableStandard;
  triggerClassName?: string;
};

export const UpdateStandard = ({ standard, triggerClassName }: UpdateStandardProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} variant="ghost" size="sm">Изменить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <UpdateStandardModal standard={standard} />
    </Modal.Content>
  </Modal>
);

const UpdateStandardModal = ({ standard }: { standard: EditableStandard }) => {
  const close = useModalClose();
  const angleOptions = useAngleOptions();
  const [name, setName] = useState(standard.name);
  const [angle, setAngle] = useState<Angle | null>(standard.angle);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const mutation = useUpdateStandard({ groupId: standard.group_id });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  const handleSubmit = () => {
    const result = buildUpdateStandardPayload(
      { name, angle },
      { name: standard.name, angle: standard.angle }
    );

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    if (!result.data) {
      return;
    }

    setFormErrors({});
    mutation.mutate({ id: standard.id, data: result.data });
  };

  return (
    <>
      <Modal.Header>{`Изменить эталон ${standard.name}`}</Modal.Header>
      <Modal.Body>
        <Input
          label="Название"
          placeholder="Пример"
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
        <Select
          label="Ракурс"
          options={angleOptions}
          value={angle ?? ""}
          placeholder="Выберите ракурс"
          onChange={(val) => {
            setAngle(val ? (val as Angle) : null);
            setFormErrors((current) => {
              const next = { ...current };
              delete next.angle;
              delete next.form;
              return next;
            });
          }}
          error={formErrors.angle ?? getFieldError(mutation.error, "angle")}
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
