import { useDisclosure } from "@/hooks/use-disclosure";
import clsx from "clsx";
import { X } from "lucide-react";
import {
  Children,
  cloneElement,
  createContext,
  MouseEventHandler,
  ReactElement,
  ReactNode,
  useContext,
  useEffect,
  useId,
  useRef,
} from "react";
import { createPortal } from "react-dom";
import s from "./modal.module.scss";

type ModalContextValue = {
  close: () => void;
};

type ModalOpenContextValue = {
  isOpen: boolean;
  open: () => void;
};

type ModalA11yContextValue = {
  titleId: string;
  bodyId: string;
};

type ModalTriggerProps = {
  onClick?: MouseEventHandler<HTMLElement>;
  "aria-expanded"?: boolean;
  "aria-haspopup"?: "dialog";
};

type ContentPlacement = "center" | "bottom";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
  "[contenteditable='true']",
].join(",");

const ModalContext = createContext<ModalContextValue>({
  close: () => {},
});

const ModalOpenContext = createContext<ModalOpenContextValue>({
  isOpen: false,
  open: () => {},
});

const ModalA11yContext = createContext<ModalA11yContextValue>({
  titleId: "",
  bodyId: "",
});

const isFocusableElement = (element: HTMLElement) => {
  if (element.getAttribute("aria-hidden") === "true") {
    return false;
  }

  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden";
};

const getFocusableElements = (root: HTMLElement) => {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(isFocusableElement);
};

const focusInitialElement = (root: HTMLElement) => {
  const [firstFocusableElement] = getFocusableElements(root);
  const target = firstFocusableElement ?? root;
  target.focus({ preventScroll: true });
};

const Root = ({ children }: { children: ReactNode }) => {
  const { isOpen, open, close } = useDisclosure();
  return (
    <ModalContext.Provider value={{ close }}>
      <ModalOpenContext.Provider value={{ isOpen, open }}>{children}</ModalOpenContext.Provider>
    </ModalContext.Provider>
  );
};

const Trigger = ({ children }: { children: ReactElement }) => {
  const { isOpen, open } = useContext(ModalOpenContext);
  const child = Children.only(children) as ReactElement<ModalTriggerProps>;

  return cloneElement(child, {
    "aria-expanded": isOpen,
    "aria-haspopup": "dialog",
    onClick: (event) => {
      child.props.onClick?.(event);

      if (!event.defaultPrevented) {
        open();
      }
    },
  });
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
  const rootRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const bodyId = useId();

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const root = rootRef.current;
    if (!root) {
      return undefined;
    }

    const previouslyFocusedElement =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const focusTimer = window.setTimeout(() => focusInitialElement(root), 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        close();
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const focusableElements = getFocusableElements(root);
      if (focusableElements.length === 0) {
        event.preventDefault();
        root.focus({ preventScroll: true });
        return;
      }

      const firstFocusableElement = focusableElements[0];
      const lastFocusableElement = focusableElements[focusableElements.length - 1];
      const activeElement = document.activeElement;

      if (event.shiftKey) {
        if (activeElement === firstFocusableElement || !root.contains(activeElement)) {
          event.preventDefault();
          lastFocusableElement.focus({ preventScroll: true });
        }
        return;
      }

      if (activeElement === lastFocusableElement) {
        event.preventDefault();
        firstFocusableElement.focus({ preventScroll: true });
      }
    };

    document.addEventListener("keydown", handleKeyDown, true);

    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener("keydown", handleKeyDown, true);

      if (previouslyFocusedElement?.isConnected) {
        previouslyFocusedElement.focus({ preventScroll: true });
      }
    };
  }, [close, isOpen]);

  if (!isOpen) {
    return null;
  }

  return createPortal(
    <div className={clsx(s.overlay, placement === "bottom" && s.overlayBottom)} onClick={close}>
      <div
        ref={rootRef}
        className={clsx(s.root, wide && s.wide, placement === "bottom" && s.rootBottom, className)}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <ModalA11yContext.Provider value={{ titleId, bodyId }}>{children}</ModalA11yContext.Provider>
      </div>
    </div>,
    document.body
  );
};

const Header = ({ children }: { children: string }) => {
  const { close } = useContext(ModalContext);
  const { titleId } = useContext(ModalA11yContext);

  return (
    <div className={s.header}>
      <span id={titleId} className={s.title}>
        {children}
      </span>
      <button type="button" className={s.close} onClick={close} aria-label="Закрыть модальное окно">
        <X />
      </button>
    </div>
  );
};

const Body = ({ children }: { children: ReactNode }) => {
  const { bodyId } = useContext(ModalA11yContext);
  return (
    <div id={bodyId} className={s.body}>
      {children}
    </div>
  );
};

const Footer = ({ children }: { children: ReactNode }) => {
  return <div className={s.footer}>{children}</div>;
};

export const useModalClose = () => useContext(ModalContext).close;
export const Modal = Object.assign(Root, { Trigger, Content, Header, Body, Footer });
