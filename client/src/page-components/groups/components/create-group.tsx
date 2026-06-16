import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { getFieldError } from "@/lib/errors";
import { Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { buildCreateGroupPayload, useCreateGroup } from "../api/create-group";
import s from "./create-group.module.scss";

export const CreateGroup = () => (
  <Modal>
    <Modal.Trigger><Button size="sm">+ New Dataset</Button></Modal.Trigger>
    <Modal.Content wide><CreateGroupModal /></Modal.Content>
  </Modal>
);

const CreateGroupModal = () => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const close = useModalClose();
  const mutation = useCreateGroup();

  useEffect(() => { if (mutation.isSuccess) close(); }, [mutation.isSuccess, close]);

  const slug = name.trim().toLowerCase().replace(/[^a-zа-яё0-9]+/gi, "-").replace(/^-+|-+$/g, "") || "new-dataset";
  const handleSubmit = () => {
    const result = buildCreateGroupPayload({ name, description });
    if (result.ok === false) { setFormErrors(result.errors); return; }
    setFormErrors({}); mutation.mutate(result.data);
  };

  return <>
    <Modal.Header>New Dataset</Modal.Header>
    <Modal.Body>
      <div className={s.hint}>Create a dataset to store reference images, standards and annotations.</div>
      <div className={s.drop}><Upload /><strong>Drop reference images here</strong><span>or create an empty dataset and upload standards later</span></div>
      <Input label="Dataset Name" placeholder="gearbox-cover" value={name} onChange={(value) => { setName(value); setFormErrors((current) => ({ ...current, name: "", form: "" })); }} error={formErrors.name ?? getFieldError(mutation.error, "name")} />
      <div className={s.urlRow}><span>/local/datasets/</span><strong>{slug}</strong><b>✓</b></div>
      <Input label="Description (optional)" value={description} onChange={(value) => { setDescription(value); setFormErrors((current) => ({ ...current, description: "", form: "" })); }} error={formErrors.description ?? getFieldError(mutation.error, "description")} />
    </Modal.Body>
    <Modal.Footer><Button variant="ghost" onClick={close}>Cancel</Button><Button disabled={mutation.isPending} onClick={handleSubmit}>Create Dataset</Button></Modal.Footer>
  </>;
};
