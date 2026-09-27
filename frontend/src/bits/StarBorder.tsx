/**
 * StarBorder — animated gradient border wrapper.
 * Adapted from React Bits (https://reactbits.dev) — MIT License.
 */
import type { CSSProperties, ElementType, ReactNode } from "react";

import "./StarBorder.css";

export interface StarBorderProps {
  as?: ElementType;
  className?: string;
  color?: string;
  speed?: string;
  thickness?: number;
  backgroundColor?: string;
  textColor?: string;
  borderColor?: string;
  style?: CSSProperties;
  children?: ReactNode;
  onClick?: () => void;
  title?: string;
}

export default function StarBorder({
  as: Component = "button",
  className = "",
  color = "white",
  speed = "6s",
  thickness = 1,
  backgroundColor = "#000000",
  textColor = "#ffffff",
  borderColor = "#222222",
  style,
  children,
  ...rest
}: StarBorderProps) {
  return (
    <Component
      className={`star-border-container ${className}`}
      style={{ padding: `${thickness}px 0`, ...style }}
      {...rest}
    >
      <div
        className="border-gradient-bottom"
        style={{
          background: `radial-gradient(circle, ${color}, transparent 10%)`,
          animationDuration: speed,
        }}
      ></div>
      <div
        className="border-gradient-top"
        style={{
          background: `radial-gradient(circle, ${color}, transparent 10%)`,
          animationDuration: speed,
        }}
      ></div>
      <div
        className="inner-content"
        style={{ background: backgroundColor, color: textColor, borderColor }}
      >
        {children}
      </div>
    </Component>
  );
}
