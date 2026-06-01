import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useDeleteGroup } from "../api/delete-group";

interface Props {
  id: string;
  name: string;
}

export const DeleteGroup = ({ id, name }: Props) => (
  <Modal>
    <Modal.Trigger>
      <Button variant="danger">Удалить</Button>
    </Modal.Trigger>
    <Modal.Content>
      <DeleteGroupModal id={id} name={name} />
    </Modal.Content>
  </Modal>
);

const DeleteGroupModal = ({ id, name }: Props) => {
  const close = useModalClose();
  const navigate = useNavigate();
  const mutation = useDeleteGroup({
    mutationConfig: {
      onSuccess: () => navigate(paths.groups()),
    },
  });

  useEffect(() => {
    if (mutation.isSuccess) {
      close();
    }
  }, [mutation.isSuccess]);

  return (
    <>
      <Modal.Header>Удалить группу</Modal.Header>
      <Modal.Body>{`Вы уверены, что хотите удалить <${name}>?`}</Modal.Body>
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
