import type { ActionType } from "../types/api";

const ACTION_ORDER: ActionType[] = ["Discrete", "Continuous", "MultiDiscrete"];

const ACTION_DESCRIPTIONS: Record<ActionType, string> = {
  Discrete: "Single-action environments such as Atari and classic control.",
  Continuous: "Continuous control tasks with vector actions.",
  MultiDiscrete: "Multiple discrete branches in the action space.",
};

interface ActionTypeSelectorProps {
  value: ActionType | "";
  options: ActionType[];
  onChange: (value: ActionType) => void;
  title: string;
  description: string;
}

export function ActionTypeSelector({
  value,
  options,
  onChange,
  title,
  description,
}: ActionTypeSelectorProps) {
  const orderedOptions = ACTION_ORDER.filter((actionType) => options.includes(actionType));

  return (
    <section className="panel p-5">
      <div className="mb-4 space-y-1">
        <div className="eyebrow">Step 1</div>
        <h2 className="text-lg font-semibold text-text">{title}</h2>
        <p className="text-sm text-muted">{description}</p>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {orderedOptions.map((actionType) => {
          const active = value === actionType;
          return (
            <button
              key={actionType}
              type="button"
              className={`rounded-lg border px-4 py-4 text-left transition ${
                active
                  ? "border-accent bg-accent/10"
                  : "border-border bg-page/60 hover:border-accent/40"
              }`}
              onClick={() => onChange(actionType)}
            >
              <div className="text-sm font-semibold text-text">{actionType}</div>
              <div className="mt-1 text-xs leading-5 text-muted">
                {ACTION_DESCRIPTIONS[actionType]}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
