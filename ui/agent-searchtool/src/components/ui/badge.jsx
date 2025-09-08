import React from "react";

export const Badge = ({ children, color = "gray" }) => {
  const colors = {
    gray: "bg-gray-100 text-gray-800",
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    blue: "bg-blue-100 text-blue-800"
  };
  return (
    <span className={`px-2 py-1 text-xs font-medium rounded ${colors[color]}`}>
      {children}
    </span>
  );
};