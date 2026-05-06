import type { ReactNode } from "react";

interface SectionCardProps {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function SectionCard({
  eyebrow,
  title,
  description,
  action,
  children,
  className,
}: SectionCardProps) {
  return (
    <section className={`panel p-5 ${className ?? ""}`}>
      <div className="mb-5 flex items-start justify-between gap-4">
        <div className="space-y-1">
          {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
          <h2 className="text-lg font-semibold text-text">{title}</h2>
          {description ? <p className="max-w-2xl text-sm text-muted">{description}</p> : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      {children}
    </section>
  );
}
