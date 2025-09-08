import React from "react";

export const Toaster = ({ message }) =>
  message ? (
    <div className="fixed bottom-4 right-4 bg-black text-white px-4 py-2 rounded">
      {message}
    </div>
  ) : null;