import { useState } from "react";

export const useToast = () => {
  const [message, setMessage] = useState("");
  const showToast = (msg) => {
    setMessage(msg);
    setTimeout(() => setMessage(""), 3000);
  };
  return { message, showToast };
};