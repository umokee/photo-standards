import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import Select from "@/components/ui/select/select";
import { useAngleOptions } from "@/constants";
import { getFieldError } from "@/lib/errors";
import type { Angle } from "@/types/contracts";
import { Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { buildCreateStandardPayload, useCreateStandard } from "../api/create-standard";
import s from "@/page-components/groups/components/create-group.module.scss";

export const CreateStandard = ({ groupId }: { groupId: string }) => (
  <Modal><Modal.Trigger><Button size="sm">+ New Reference</Button></Modal.Trigger><Modal.Content wide><CreateStandardModal groupId={groupId} /></Modal.Content></Modal>
);

const CreateStandardModal = ({ groupId }: { groupId: string }) => {
  const close = useModalClose();
  const angleOptions = useAngleOptions();
  const [name, setName] = useState("");
  const [angle, setAngle] = useState<Angle | null>(null);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const mutation = useCreateStandard();
  useEffect(() => { if (mutation.isSuccess) close(); }, [mutation.isSuccess, close]);
  const handleSubmit = () => { const result = buildCreateStandardPayload({ groupId, name, angle }); if (result.ok === false) { setFormErrors(result.errors); return; } setFormErrors({}); mutation.mutate(result.data); };
  return <><Modal.Header>New Reference</Modal.Header><Modal.Body><div className={s.hint}>Create a standard view for this dataset.</div><div className={s.drop}><Upload /><strong>Drop reference images</strong><span>После создания открой эталон и добавь изображения.</span></div><Input label="Reference Name" placeholder="front view" value={name} onChange={(value) => { setName(value); setFormErrors((current) => ({ ...current, name: "", form: "" })); }} error={formErrors.name ?? getFieldError(mutation.error, "name")} /><Select label="View angle" options={angleOptions} value={angle ?? ""} placeholder="Select angle" onChange={(val) => { setAngle(val ? (val as Angle) : null); setFormErrors((current) => ({ ...current, angle: "", form: "" })); }} error={formErrors.angle ?? getFieldError(mutation.error, "angle")} /></Modal.Body><Modal.Footer><Button variant="ghost" onClick={close}>Cancel</Button><Button disabled={mutation.isPending} onClick={handleSubmit}>Create Reference</Button></Modal.Footer></>;
};
