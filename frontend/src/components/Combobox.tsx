import { useMemo, useState } from "react";

interface ComboboxProps {
  label: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export default function Combobox({ label, options, value, onChange, placeholder }: ComboboxProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const filtered = useMemo(() => {
    if (!query) return options.slice(0, 50);
    const q = query.toLowerCase();
    return options.filter((o) => o.toLowerCase().includes(q)).slice(0, 50);
  }, [options, query]);

  return (
    <div className="relative">
      <label className="mb-1 block text-sm font-medium text-neutral-600 dark:text-neutral-300">
        {label}
      </label>
      <input
        type="text"
        value={open ? query : value}
        placeholder={placeholder ?? "Search a player..."}
        onFocus={() => {
          setOpen(true);
          setQuery("");
        }}
        onChange={(e) => setQuery(e.target.value)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface dark:bg-surface-dark
                   px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1 dark:focus:ring-series1-dark"
      />
      {open && (
        <ul
          role="listbox"
          className="absolute z-10 mt-1 max-h-56 w-full overflow-auto rounded-lg border border-neutral-200
                     dark:border-neutral-700 bg-surface dark:bg-surface-dark shadow-lg"
        >
          {filtered.length === 0 && (
            <li className="px-3 py-2 text-sm text-neutral-500">No players found</li>
          )}
          {filtered.map((option) => (
            <li
              key={option}
              role="option"
              aria-selected={option === value}
              onMouseDown={() => {
                onChange(option);
                setOpen(false);
              }}
              className={`cursor-pointer px-3 py-2 text-sm hover:bg-neutral-100 dark:hover:bg-neutral-800 ${
                option === value ? "font-medium" : ""
              }`}
            >
              {option}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
