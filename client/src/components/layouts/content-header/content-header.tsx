import { createContext, ReactNode, useContext } from "react";
import s from "./content-header.module.scss";

interface TopProps {
  children?: ReactNode;
  title: string;
  subtitles?: string[];
  meta?: string[];
}

const InsideRootContext = createContext(false);
const useInsideRoot = (componentName: string) => {
  const insideRoot = useContext(InsideRootContext);

  if (!insideRoot) {
    throw new Error(`${componentName} должен быть использован внутри ContentHeader`);
  }
};

const InsideTopContext = createContext(false);
const useInsideTop = (componentName: string) => {
  const insideTop = useContext(InsideTopContext);

  if (!insideTop) {
    throw new Error(`${componentName} должен быть использован внутри ContentHeader.Top`);
  }
};

const Actions = ({ children }: { children: ReactNode }) => {
  useInsideTop("ContentHeader.Actions");
  return <div className={s.actions}>{children}</div>;
};

const Top = ({ children, title, subtitles = [], meta = [] }: TopProps) => {
  useInsideRoot("ContentHeader.Top");

  return (
    <InsideTopContext.Provider value={true}>
      <div className={s.top}>
        <div className={s.textBlock}>
          <span className={s.title}>{title}</span>
          {subtitles.map((subtitle) => (
            <span key={subtitle} className={s.sub}>
              {subtitle}
            </span>
          ))}
        </div>
        {children}
      </div>

      {meta.length > 0 && (
        <div className={s.meta}>
          {meta.map((item) => (
            <span key={item} className={s.stat}>
              {item}
            </span>
          ))}
        </div>
      )}
    </InsideTopContext.Provider>
  );
};

const Root = ({ children }: { children: ReactNode }) => {
  return (
    <InsideRootContext.Provider value={true}>
      <div className={s.root}>{children}</div>
    </InsideRootContext.Provider>
  );
};

export const ContentHeader = Object.assign(Root, {
  Top,
  Actions,
});
