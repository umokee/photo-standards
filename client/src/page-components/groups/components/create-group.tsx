import { paths } from "@/app/paths";
import p from "@/app/routes/platform-pages.module.scss";
import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import { CheckCircle2, Database, ImagePlus, Layers3, Plus, Rocket, Tags } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { buildCreateGroupPayload, useCreateGroup } from "../api/create-group";

export const CreateGroup = () => (
  <Modal>
    <Modal.Trigger>
      <Button icon={Plus} size="sm">
        New Dataset
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
      <Modal.Header>New Dataset</Modal.Header>
      <Modal.Body>
        <div className={p.modalHeroCard}>
          <div className={p.modalHeroIcon}><Database /></div>
          <div>
            <strong>Dataset изделия</strong>
            <span>Контейнер для reference views, классов, полигонов, обучения и проверок.</span>
          </div>
        </div>

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

        <div className={p.modalWorkflowGrid}>
          <WorkflowStep icon={ImagePlus} title="Reference" text="Добавь эталонные ракурсы" />
          <WorkflowStep icon={Tags} title="Classes" text="Создай классы деталей" />
          <WorkflowStep icon={Layers3} title="Annotate" text="Разметь контрольные зоны" />
          <WorkflowStep icon={Rocket} title="Inspect" text="Запускай проверки" />
        </div>

        {formErrors.form ? <div className={p.modalError}>{formErrors.form}</div> : null}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Cancel
        </Button>

        <Button icon={CheckCircle2} disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Creating..." : "Create Dataset"}
        </Button>
      </Modal.Footer>
    </>
  );
};

function WorkflowStep({ icon: Icon, title, text }: { icon: typeof ImagePlus; title: string; text: string }) {
  return (
    <div className={p.modalWorkflowStep}>
      <Icon />
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}
