"use client"

import * as React from "react"
import * as TooltipPrimitive from "@radix-ui/react-tooltip"
import { cn } from '../../lib/utils.js';

const TooltipProvider = TooltipPrimitive.Provider

const Tooltip = ({ children, className, ...props }) => (
  <TooltipPrimitive.Root {...props}>
    {children}
  </TooltipPrimitive.Root>
)

const TooltipTrigger = TooltipPrimitive.Trigger
const TooltipContent = ({ className, ...props }) => (
  <TooltipPrimitive.Content
    className={cn(
      "rounded-md bg-gray-900 p-2 text-white text-sm shadow-md",
      className
    )}
    {...props}
  />
)

export { Tooltip, TooltipProvider, TooltipTrigger, TooltipContent }
