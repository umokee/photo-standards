import { useDisclosure } from "@/hooks/use-disclosure";
import clsx from "clsx";
import { X } from "lucide-react";
import { createContext, ReactElement, ReactNode, useContext } from "react";
import { createPortal } from "react-dom";
import s from "./modal.module.scss";

type ModalContenxValue = {
  close: () => void;
};

type ModalOpenContextValue = {
  isOpen: boolean;
  open: () => void;
};

type ContentPlacement = "center" | "bottom";

const ModalContext = createContext<ModalContenxValue>({
  close: () => {},
});

const ModalOpenContext = createContext<ModalOpenContextValue>({
  isOpen: false,
  open: () => {},
});

const Root = ({ children }: { children: ReactNode }) => {
  const { isOpen, open, close } = useDisclosure();
  return (
    <ModalContext.Provider value={{ close }}>
      <ModalOpenContext.Provider value={{ isOpen, open }}>{children}</ModalOpenContext.Provider>
    </ModalContext.Provider>
  );
};

const Trigger = ({ children }: { children: ReactElement }) => {
  const { open } = useContext(ModalOpenContext);
  return (
    <span onClick={open} style={{ display: "contents" }}>
      {children}
    </span>
  );
};

const Content = ({
  children,
  wide = false,
  className,
  placement = "center",
}: {
  children: ReactNode;
  wide?: boolean;
  className?: string;
  placement?: ContentPlacement;
}) => {
  const { isOpen } = useContext(ModalOpenContext);
  const { close } = useContext(ModalContext);

  if (!isOpen) {
    return null;
  }

  return createPortal(
    <div className={clsx(s.overlay, placement === "bottom" && s.overlayBottom)} onClick={close}>
      <div
        className={clsx(s.root, wide && s.wide, placement === "bottom" && s.rootBottom, className)}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>,
    document.body
  );
};

const Header = ({ children }: { children: string }) => {
  const { close } = useContext(ModalContext);

  return (
    <div className={s.header}>
      <span className={s.title}>{children}</span>
      <button type="button" className={s.close} onClick={close} aria-label="Закрыть модальное окно">
        <X />
      </button>
    </div>
  );
};

const Body = ({ children }: { children: ReactNode }) => {
  return <div className={s.body}>{children}</div>;
};

const Footer = ({ children }: { children: ReactNode }) => {
  return <div className={s.footer}>{children}</div>;
};

export const useModalClose = () => useContext(ModalContext).close;
export const Modal = Object.assign(Root, { Trigger, Content, Header, Body, Footer });
