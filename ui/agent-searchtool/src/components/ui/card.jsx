"use client"

import * as React from "react";
import { cn } from '../../lib/utils.js';


const Card = ({ className, ...props }) => (
  <div className={cn("rounded-lg border bg-card p-4", className)} {...props} />
)

const CardHeader = ({ className, ...props }) => (
  <div className={cn("flex flex-col space-y-1.5 p-4", className)} {...props} />
)

const CardContent = ({ className, ...props }) => (
  <div className={cn("p-4", className)} {...props} />
)

const CardTitle = ({ className, ...props }) => (
  <h3 className={cn("text-lg font-semibold", className)} {...props} />
)

export { Card, CardContent, CardHeader, CardTitle };

