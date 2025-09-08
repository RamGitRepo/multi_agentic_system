import React, { useState } from "react";

export const DropdownMenu = ({ trigger, items }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative inline-block">
      <span onClick={() => setOpen(!open)}>{trigger}</span>
      {open && (
        <div className="absolute mt-2 w-48 bg-white shadow-lg border rounded">
          {items.map((item, idx) => (
            <div
              key={idx}
              className="px-4 py-2 hover:bg-gray-100 cursor-pointer"
              onClick={item.onClick}
            >
              {item.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};