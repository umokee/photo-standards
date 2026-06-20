import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { Trash2 } from "lucide-react";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useDeleteStandard } from "../api/delete-standard";

interface Props {
  groupId: string;
  id: string;
  name: string;
}

type DeleteStandardTriggerProps = Props & {
  triggerClassName?: string;
};

export const DeleteStandard = ({ groupId, id, name, triggerClassName }: DeleteStandardTriggerProps) => (
  <Modal>
    <Modal.Trigger>
      <Button className={triggerClassName} variant="danger" size="sm">Удалить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <DeleteStandardModal groupId={groupId} id={id} name={name} />
    </Modal.Content>
  </Modal>
);

const DeleteStandardModal = ({ groupId, id, name }: Props) => {
  const close = useModalClose();
  const navigate = useNavigate();
  const mutation = useDeleteStandard({
    groupId,
    mutationConfig: {
      onSuccess: () => navigate(paths.groupDetail(groupId)),
    },
  });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  return (
    <>
      <Modal.Header>Удалить эталон</Modal.Header>
      <Modal.Body>{`Вы уверены, что хотите удалить эталон «${name}»?`}</Modal.Body>
      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button variant="danger" disabled={mutation.isPending} onClick={() => mutation.mutate(id)}>
          Удалить
        </Button>
      </Modal.Footer>
    </>
  );
};
