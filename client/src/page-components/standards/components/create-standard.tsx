import p from "./standard-modal.module.scss";
import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import { useAngleOptions } from "@/constants";
import { getFieldError } from "@/lib/errors";
import type { Angle } from "@/types/contracts";
import { CheckCircle2, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { buildCreateStandardPayload, useCreateStandard } from "../api/create-standard";

type CreateStandardProps = {
  groupId: string;
  triggerClassName?: string;
};

export const CreateStandard = ({ groupId, triggerClassName }: CreateStandardProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} icon={Plus}>Новый эталон</Button>
    </Modal.Trigger>
    <Modal.Content wide>
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
  }, [mutation.isSuccess, close]);

  const handleSubmit = () => {
    const result = buildCreateStandardPayload({ groupId, name, angle });

    if (result.ok === false) {
      setFormErrors(result.errors);
      return;
    }

    setFormErrors({});
    mutation.mutate(result.data);
  };

  return (
    <>
      <Modal.Header>Новый эталон</Modal.Header>
      <Modal.Body>
        <div className={p.modalFormGrid}>
          <Input
            label="Название"
            placeholder="Например: Вид сверху"
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
        </div>

        {formErrors.form ? <div className={p.modalError}>{formErrors.form}</div> : null}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button icon={CheckCircle2} disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Создание..." : "Создать эталон"}
        </Button>
      </Modal.Footer>
    </>
  );
};

