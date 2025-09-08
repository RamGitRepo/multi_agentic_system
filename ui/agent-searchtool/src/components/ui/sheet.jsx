"use client"

import * as React from "react"
import * as SheetPrimitive from "@radix-ui/react-dialog"
import { cn } from '../../lib/utils.js';

const Sheet = SheetPrimitive.Root
const SheetTrigger = SheetPrimitive.Trigger
const SheetContent = ({ className, ...props }) => (
  <SheetPrimitive.Content
    className={cn(
      "fixed right-0 top-0 h-full w-full max-w-sm bg-white p-6 shadow-lg",
      className
    )}
    {...props}
  />
)
const SheetHeader = ({ className, ...props }) => (
  <div className={cn("flex flex-col space-y-2", className)} {...props} />
)
const SheetTitle = ({ className, ...props }) => (
  <h2 className={cn("text-lg font-semibold", className)} {...props} />
)
const SheetDescription = ({ className, ...props }) => (
  <p className={cn("text-sm text-muted-foreground", className)} {...props} />
)

export { Sheet, SheetTrigger, SheetContent, SheetHeader, SheetTitle, SheetDescription }
