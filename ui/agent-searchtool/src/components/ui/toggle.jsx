import React from "react";

export const Toggle = ({ checked, onChange }) => (
  <label className="inline-flex items-center cursor-pointer">
    <input type="checkbox" checked={checked} onChange={onChange} className="hidden" />
    <span
      className={`w-10 h-6 flex items-center bg-gray-300 rounded-full p-1 transition ${
        checked ? "bg-green-500" : ""
      }`}
    >
      <span
        className={`bg-white w-4 h-4 rounded-full shadow-md transform transition ${
          checked ? "translate-x-4" : ""
        }`}
      />
    </span>
  </label>
);