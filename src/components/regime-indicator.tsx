interface RegimeIndicatorProps {
  regime: "bullish" | "bearish" | "neutral"
}

export function RegimeIndicator({ regime }: RegimeIndicatorProps) {
  const getColor = () => {
    switch (regime) {
      case "bullish":
        return {
          bg: "bg-green-500/10",
          border: "border-green-500/30",
          text: "text-green-400",
          dot: "bg-green-400",
          label: "Bull Market",
        }
      case "bearish":
        return {
          bg: "bg-red-500/10",
          border: "border-red-500/30",
          text: "text-red-400",
          dot: "bg-red-400",
          label: "Bear Market",
        }
      case "neutral":
        return {
          bg: "bg-yellow-500/10",
          border: "border-yellow-500/30",
          text: "text-yellow-400",
          dot: "bg-yellow-400",
          label: "Crab Market",
        }
      default:
        return {
          bg: "bg-green-500/10",
          border: "border-green-500/30",
          text: "text-green-400",
          dot: "bg-green-400",
          label: "Bull Market",
        }
    }
  }

  const color = getColor()

  return (
    <div className={`inline-flex items-center rounded-full border px-3 py-1 text-sm ${color.border} ${color.bg}`}>
      <span className={`mr-2 h-2 w-2 rounded-full ${color.dot}`}></span>
      <span className={`font-medium ${color.text}`}>{color.label}</span>
    </div>
  )
}
