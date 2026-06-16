import p from "@/app/routes/platform-pages.module.scss";
import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import { useAngleOptions } from "@/constants";
import { getFieldError } from "@/lib/errors";
import type { Angle } from "@/types/contracts";
import { Camera, CheckCircle2, Compass, ImagePlus, Layers3, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { buildCreateStandardPayload, useCreateStandard } from "../api/create-standard";

export const CreateStandard = ({ groupId }: { groupId: string }) => (
  <Modal>
    <Modal.Trigger>
      <Button icon={Plus}>New Reference</Button>
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
      <Modal.Header>New Reference</Modal.Header>
      <Modal.Body>
        <div className={p.modalHeroCard}>
          <div className={p.modalHeroIcon}><ImagePlus /></div>
          <div>
            <strong>Эталонный ракурс</strong>
            <span>Reference view хранит изображения и полигоны для переноса на проверяемый кадр.</span>
          </div>
        </div>

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

        <div className={p.modalWorkflowGrid}>
          <WorkflowStep icon={Camera} title="Capture" text="Загрузи кадры эталона" />
          <WorkflowStep icon={Compass} title="Angle" text="Зафиксируй ракурс" />
          <WorkflowStep icon={Layers3} title="Markup" text="Нарисуй зоны деталей" />
        </div>

        {formErrors.form ? <div className={p.modalError}>{formErrors.form}</div> : null}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Cancel
        </Button>
        <Button icon={CheckCircle2} disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Creating..." : "Create Reference"}
        </Button>
      </Modal.Footer>
    </>
  );
};

function WorkflowStep({ icon: Icon, title, text }: { icon: typeof Camera; title: string; text: string }) {
  return (
    <div className={p.modalWorkflowStep}>
      <Icon />
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}
